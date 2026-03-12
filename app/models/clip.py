from functools import lru_cache
import os
import logging

import torch
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_images

from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model = model.to(device)

@lru_cache(maxsize=128)
def vectorize_image(image_path: str, root_dir: str = "uploads") -> torch.Tensor:
    with Image.open(os.path.join(root_dir, image_path)) as image:
        inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        return model.get_image_features(**inputs).pooler_output


def vectorize_images(image_paths: list[str], root_dir: str = "uploads") -> torch.FloatTensor:
    logger.info(f"Vectorizing {len(image_paths)} images using {device}")
    images = load_images([os.path.join(root_dir, img) for img in image_paths])
    logger.info(f"Loaded {len(images)} images")
    inputs = processor(images=images, return_tensors="pt")
    logger.info(f"Processed {len(image_paths)} images")
    with torch.inference_mode():
        features = model.get_image_features(**inputs).pooler_output
        logger.info(f"Vectorized {len(image_paths)} images")
    return features.cpu()

def vectorize_text(text: str):
    inputs = processor(text=text, return_tensors="pt")
    with torch.inference_mode():
        return model.get_text_features(**inputs).pooler_output
    
if __name__ == "__main__":
    print(f"Testing vectorization using {device}...")
