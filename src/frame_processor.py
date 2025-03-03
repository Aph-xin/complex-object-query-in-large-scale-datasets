import PIL.Image
from encoder.owl_reranker import OwlPredictor, OwlDecodeOutput
import numpy as np
from typing import Tuple, List
from logger import Logger
from pymilvus import Collection 
from db_manager import MilvusManager


class FrameProcessor:
    """Image frame processing and encoding class"""
    def __init__(self, predictor: OwlPredictor):
        self.predictor = predictor

    @staticmethod
    def load_image(image_path: str) -> PIL.Image.Image:
        """Load and convert a frame to RGB format"""
        image = PIL.Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def encode_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a frame and return embeddings and bounding boxes"""
        image = self.load_image(image_path)
        output = self.predictor.image_encoder_milvus(image=image, pad_square=False)
        image_embeds = output.image_class_embeds_aug.squeeze().cpu().detach().numpy()
        pred_boxes = output.pred_boxes.squeeze().cpu().detach().numpy()
        return image_embeds, pred_boxes

    def encode_and_store(self, image_path: str, collection: Collection, logger: Logger):
        """Encode a frame and store it in Milvus"""
        image_embeds, pred_boxes = self.encode_image(image_path)
        data = [
            {"image_embeds": embed.tolist(), "pred_boxes": box.tolist(), "image_id": image_path}
            for embed, box in zip(image_embeds, pred_boxes)
        ]
        MilvusManager().insert_data(collection, data, logger)

    def encode_query(self, prompt: str) -> OwlDecodeOutput:
        """Encode a text query"""
        prompt = prompt.strip("][()")
        texts = prompt.split(',')
        return self.predictor.encode_text([texts])