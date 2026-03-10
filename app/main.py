import asyncio
import os

from aiofiles.os import listdir, remove, rmdir
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    UploadFile,
    status,
    File,
    BackgroundTasks,
)
from typing import Annotated, List, Optional

from pydantic import BaseModel
from app.rag.vector_service import VectorDatabaseType, VectorServiceFactory
from app.upload import save_file
import uvicorn

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


ALLOWED_CONTENT_TYPES = {
    "image/jpg",
    "image/jpeg",
    "image/png",
}


class SearchRequest(BaseModel):  # new
    query: str
    top_k: int = 5

vector_service = VectorServiceFactory.create(
    db_type=VectorDatabaseType.QDRANT,
    collection_name="images",
    host="qdrant"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # executor.shutdown(wait=True)
    try:
        files = await listdir("uploads")
        for file in files:
            await remove(os.path.join("uploads", file))
        await rmdir("uploads")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"An error occurred while deleting files - Error: {e}")

    # clear the vector database on shutdown
    try:
        vector_service.clear_all()
    except Exception as e:
        print(f"Failed to clear vectors on shutdown - {e}")


app = FastAPI(lifespan=lifespan)


def paginate(skip: int = 0, limit: int = 5):
    return {"skip": skip, "limit": limit}


@app.post("/upload")
async def file_upload_controller(
    files: Annotated[list[UploadFile], File(description="Uploaded images")],
    bg_image_processor: BackgroundTasks,
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

        bg_image_processor.add_task(vector_service.process_images, filenames)

    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"message": "Files uploaded successfully"}


@app.get("/images")
async def get_images_controller(
    pagination: dict = Depends(paginate),
    file_paths: Optional[List[str]] = Query(
        None,
        alias="file_paths",
        description="specific filenames to return; paging is applied to the result",
    ),
):
    base = "uploads"
    try:
        if file_paths:
            # build list of the requested files that actually exist
            images = [fn for fn in file_paths if os.path.exists(os.path.join(base, fn))]
        else:
            # default behaviour: list the directory
            images = await listdir(base)
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


@app.post("/search")
def search_similar_images_controller(request: SearchRequest):
    try:
        query_vector = vector_service.vectorize_query(request.query)
        results = vector_service.search_similar(
            query_vector=query_vector, top_k=request.top_k
        )
        filenames = [result.payload["filename"] for result in results]
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred during search - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"results": filenames}


if __name__ == "__main__":
    uvicorn.run(app)
