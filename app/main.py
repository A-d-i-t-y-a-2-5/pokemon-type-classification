import asyncio
import os

from aiofiles.os import listdir, remove, rmdir
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import Depends, FastAPI, HTTPException, UploadFile, status, File
from typing import Annotated
from upload import save_file
import uvicorn

from models.clip import vectorize_images


ALLOWED_CONTENT_TYPES = {
    "image/jpg",
    "image/jpeg",
    "image/png",
}


executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    executor.shutdown(wait=True)
    try:
        files = await listdir("uploads")
        for file in files:
            await remove(os.path.join("uploads", file))
        await rmdir("uploads")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"An error occurred while deleting files - Error: {e}")


app = FastAPI(lifespan=lifespan)


def paginate(skip: int = 0, limit: int = 5):
    return {"skip": skip, "limit": limit}


@app.post("/upload")
async def file_upload_controller(
    files: Annotated[list[UploadFile], File(description="Uploaded images")],
):
    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                detail=f"{file.filename} is not supported. Only images (JPEG, PNG) are allowed",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
    try:
        filenames = []
        for file in files:
            await save_file(file)
            filenames.append(os.path.basename(file.filename))

        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(executor, vectorize_images, filenames)
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {
        "message": "Files uploaded successfully",
        "features_rows": features.shape[0],
        "features_columns": features.shape[1],
    }


@app.get("/images")
async def get_images_controller(pagination: dict = Depends(paginate)):
    try:
        images = await listdir("uploads")
        paginated = images[
            pagination["skip"] : pagination["skip"] + pagination["limit"]
        ]
    except FileNotFoundError:
        raise HTTPException(
            detail="Uploads directory not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while retrieving images - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"images": paginated, "total": len(images)}


if __name__ == "__main__":
    uvicorn.run(app)
