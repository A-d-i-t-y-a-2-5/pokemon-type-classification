from functools import lru_cache
import os

import torch
from transformers import AutoProcessor, CLIPModel

from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


@lru_cache(maxsize=128)
def vectorize_image(image_path: str) -> torch.Tensor:
    image = Image.open(os.path.join("uploads", image_path))
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        return model.get_image_features(**inputs).pooler_output


def vectorize_images(image_paths: list[str]) -> torch.FloatTensor:
    images = [Image.open(os.path.join("uploads", img)) for img in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        return model.get_image_features(**inputs).pooler_output
