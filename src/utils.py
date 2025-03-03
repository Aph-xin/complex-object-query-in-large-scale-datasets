import os
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np


def plot_and_save_dino_results(result, save_dir):
    """
    Draw bounding boxes of DINO re-ranking results and save the image.
    :param result: Result from unique_dino_reranker, in the format [(image_path, output_rerank), ...]
    :param save_dir: Directory to save the images
    """
    for image_path, output_rerank in result:
        boxes_filt = output_rerank["boxes"]
        labels = output_rerank["labels"]
        scores = output_rerank["scores"]
        max_score = max(scores) if scores else 0.0

        # Load the original image
        image_pil = Image.open(image_path).convert("RGB")
        H, W = image_pil.size[1], image_pil.size[0]

        # Prepare drawing
        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        device = boxes_filt.device
        size_tensor = torch.Tensor([W, H, W, H]).to(device)

        # Draw bounding boxes and labels
        for box, label in zip(boxes_filt, labels):
            box = box * size_tensor
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")
            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        # Save the image
        os.makedirs(save_dir, exist_ok=True)
        image_filename = os.path.basename(image_path).replace(" ", "_")
        image_save_path = os.path.join(save_dir, f"{max_score:.2f}_{image_filename}_result.png")
        image_pil.save(image_save_path)
        print(f"Saved image with boxes to {image_save_path}")


def format_prompt(prompt: str, model_type: str = "owl") -> str:
    """
    Format the prompt based on the model type.
    :param prompt: The input prompt string
    :param model_type: 'owl' or 'dino', determines the output format
    :return: The formatted prompt
    """
    # Remove possible square brackets (to handle inconsistent input)
    cleaned_prompt = prompt.strip("[]")
    
    if model_type == "owl":
        # OWL requires square brackets
        return f"[{cleaned_prompt}]"
    elif model_type == "dino":
        # DINO does not use square brackets
        return cleaned_prompt
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'owl' or 'dino'.")
