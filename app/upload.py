import os
import aiofiles
from aiofiles.os import makedirs
import asyncio
import io
from fastapi import UploadFile
from PIL import Image
from typing import Optional

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes
DEFAULT_SHORT_SIDE = 512

def _requires_resize(image_bytes: bytes, target_short_side: int) -> bool:
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size
    return min(width, height) > target_short_side

async def save_file(file: UploadFile, resize_short_side: Optional[int] = DEFAULT_SHORT_SIDE) -> str:
    await makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", os.path.basename(file.filename))
    contents = await file.read()
    if resize_short_side and resize_short_side > 0:
        needs_resize = await asyncio.to_thread(
            _requires_resize, contents, resize_short_side
        )
        if needs_resize:
            contents = await asyncio.to_thread(
                _resize_image, contents, resize_short_side, file.content_type
            )
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(contents)
    return filepath

def _resize_image(image_bytes: bytes, target_short_side: int, content_type: str) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size
        current_short = min(width, height)
        if current_short <= 0 or current_short <= target_short_side:
            return image_bytes
        scale = target_short_side / current_short
        new_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        img = img.resize(new_size, Image.LANCZOS)
        fmt = img.format or ("PNG" if content_type == "image/png" else "JPEG")
        output = io.BytesIO()
        img.save(output, format=fmt)
    return output.getvalue()