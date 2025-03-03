import time
import os
from typing import List, Tuple, Dict
import torch
import numpy as np
import PIL.Image
from encoder.owl_reranker import OwlPredictor, OwlDecodeOutput
from encoder.owl_drawing import draw_owl_output
from groundingdino.datasets.transforms import T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.vl_utils import create_positive_map_from_span
from logger import Logger
from frame_processor import FrameProcessor
from pymilvus import Collection


def timeit(logger: Logger):
    """Decorator: Record method execution time"""
    def decorator(method):
        def timed(*args, **kw):
            start_time = time.time()
            result = method(*args, **kw)
            elapsed_time = time.time() - start_time
            logger.info(f"{method.__name__} executed in {elapsed_time:.2f} seconds")
            return result
        return timed
    return decorator


class SearchEngine:
    """Search engine"""
    def __init__(self, predictor: OwlPredictor, logger: Logger):
        self.predictor = predictor
        self.logger = logger
        # Dynamically apply the timeit decorator
        self.unique_dino_reranker = timeit(self.logger)(self.unique_dino_reranker)
        self.display_top_results = timeit(self.logger)(self.display_top_results)

    def search_similar_images(self, text_output: OwlDecodeOutput, collection: Collection, top_n: int) -> List:
        """Search similar frames"""
        text_embeds = text_output.text_embeds.squeeze().cpu().detach().numpy()
        text_embeds = text_embeds[np.newaxis, :] if len(text_embeds.shape) == 1 else text_embeds
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = [
            collection.search(
                [embed.astype(float)], anns_field="image_embeds", param=search_params,
                limit=top_n, output_fields=["pred_boxes", "image_id"]
            ) for embed in text_embeds
        ]
        return results

    def object_rerank(self, image_path: str, prompts: str, text_output: OwlDecodeOutput, threshold: float) -> Tuple[OwlDecodeOutput, float]:
        """Re-rank image using OWL Predictor"""
        image = FrameProcessor.load_image(image_path)
        output = self.predictor.predict(
            image=image, text=prompts, text_encodings=text_output, threshold=threshold, pad_square=False
        )
        max_score = output.scores.max().item() if output.scores.numel() > 0 else 0.0
        return output, max_score

    def _load_dino_model(self, config_file: str, checkpoint_path: str, cpu_only: bool = False):
        """Load DINO model"""
        args = SLConfig.fromfile(config_file)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        return model.eval()

    def _load_dino_image(self, image_path: str):
        """Load and transform frame for DINO"""
        image_pil = PIL.Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image

    def dino_rerank(self, config_file: str, checkpoint_path: str, image_path: str, text_prompt: str, 
                    box_threshold: float = 0.3, text_threshold: float = 0.1, cpu_only: bool = False) -> Tuple[Dict, float]:
        """
        Perform re-ranking using Grounding DINO, output results compatible with OWL format.
        """
        image_pil, image = self._load_dino_image(image_path)
        model = self._load_dino_model(config_file, checkpoint_path, cpu_only)
        text_prompt = text_prompt.lower().strip()
        if not text_prompt.endswith("."):
            text_prompt += "."
        device = "cuda" if not cpu_only else "cpu"
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[text_prompt])
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        scores = logits.max(dim=1)[0]
        filt_mask = scores > box_threshold
        boxes_filt = boxes[filt_mask]
        scores_filt = scores[filt_mask].tolist()
        pred_phrases = [f"{text_prompt} ({score:.2f})" for score in scores_filt]
        output = {
            "boxes": boxes_filt,
            "labels": pred_phrases,
            "scores": scores_filt,
        }
        max_score = max(scores_filt) if scores_filt else 0.0
        return output, max_score

    def save_image_with_boxes(self, image_with_boxes: PIL.Image.Image, data_name: str, image_id: str, max_score: float,
                              model_name: str, threshold: float, save_dir: str):
        """Save image with bounding boxes"""
        os.makedirs(save_dir, exist_ok=True)
        base_image_id = os.path.splitext(os.path.basename(image_id))[0]
        file_name = f"{data_name}_{max_score:.2f}_{model_name}_thresh{threshold}_{base_image_id}.jpg"
        file_path = os.path.join(save_dir, file_name)
        image_with_boxes.save(file_path)
        self.logger.info(f"Frame saved to {file_path}")

    def unique_dino_reranker(self, config_file: str, checkpoint_path: str, results: List, prompts: str, 
                             box_threshold: float, text_threshold: float, top_k: int, save_dir: str) -> List[Tuple[str, Dict]]:
        """Perform unique rerank using DINO and save log"""
        rerank_results = []
        rerank_start_time = time.time()
        for result_group in results:
            for result in result_group:
                for match in result:
                    image_id = match.entity.get("image_id")
                    output_rerank, max_score = self.dino_rerank(
                        config_file, checkpoint_path, image_id, prompts, box_threshold, text_threshold
                    )
                    rerank_results.append((image_id, max_score, output_rerank))
        rerank_time = time.time() - rerank_start_time
        self.logger.info(f"Rerank processing time: {rerank_time:.2f} seconds")
        unique_results = {}
        for image_id, score, output_rerank in rerank_results:
            if image_id not in unique_results or score > unique_results[image_id][0]:
                unique_results[image_id] = (score, output_rerank)
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        log_entries = [f"Image ID: {image_id}, Score: {score}" for image_id, (score, _) in sorted_results]
        os.makedirs(save_dir, exist_ok=True)
        log_file_path = os.path.join(save_dir, "top_k_rerank_log.txt")
        with open(log_file_path, "w") as log_file:
            log_file.write("\n".join(log_entries))
        self.logger.info(f"Top {top_k} unique re-rank results log saved to {log_file_path}")
        return [(image_id, output_rerank) for image_id, (_, output_rerank) in sorted_results]

    def display_top_results(self, results: List, prompts: str, text_output: OwlDecodeOutput, data_name: str,
                            model_prefix: str, threshold: float, top_k: int, save_dir: str):
        """Display and save top k reranked results (no DINO)"""
        rerank_results = []
        for result_group in results:
            for result in result_group:
                for match in result:
                    image_id = match.entity.get("image_id")
                    output_rerank, max_score = self.object_rerank(image_id, prompts, text_output, threshold)
                    rerank_results.append((image_id, max_score, output_rerank))
        rerank_results.sort(key=lambda x: x[1], reverse=True)
        seen_image_ids = set()
        saved_count = 0
        for image_id, max_score, output_rerank in rerank_results:
            if image_id not in seen_image_ids:
                seen_image_ids.add(image_id)
                image_with_boxes = draw_owl_output(PIL.Image.open(image_id), output_rerank, prompts)
                self.save_image_with_boxes(image_with_boxes, data_name, image_id, max_score, model_prefix, threshold, save_dir)
                saved_count += 1
                if saved_count >= top_k:
                    break
        self.logger.info(f"Successfully saved {saved_count} unique frames.")
