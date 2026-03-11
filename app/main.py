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
from app.rag.vector_service import VectorServiceFactory
from app.rag.vector_service_async import AsyncVectorServiceFactory
from app.rag.constants import VectorDatabaseType
from app.upload import save_file
import uvicorn

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


ALLOWED_CONTENT_TYPES = {
    "image/jpg",
    "image/jpeg",
    "image/png",
}


class SearchRequest(BaseModel):  # new
    query: str
    top_k: int = 5


vector_service = None
vectorization_queue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_service, vectorization_queue
    vector_service = await AsyncVectorServiceFactory.create(
        db_type=VectorDatabaseType.AQDRANT, collection_name="images", host="qdrant"
    )
    vectorization_queue = []
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
        await vector_service.clear_all()
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
        for file in files:
            await save_file(file)
            vectorization_queue.append(os.path.basename(file.filename))

    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"message": "Files uploaded successfully"}

@app.post("/process")
async def process_images_controller(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the vectorization of files that have been uploaded.
    """
    
    if not vectorization_queue:
        return {"message": "No pending files to process."}

    files_to_process = list(vectorization_queue)
    vectorization_queue.clear()  # Clear the list after copying
    
    background_tasks.add_task(vector_service.process_images, files_to_process)
    
    logging.info(f"Triggered processing for {len(files_to_process)} files.")
    return {"message": f"Processing initiated for {len(files_to_process)} files."}

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
async def search_similar_images_controller(request: SearchRequest):
    try:
        loop = asyncio.get_running_loop()
        query_vector = await loop.run_in_executor(
            None, vector_service.vectorize_query, request.query
        )
        results = await vector_service.search_similar(
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
