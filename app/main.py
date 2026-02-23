from fastapi import FastAPI, HTTPException, UploadFile, status, File
from typing import Annotated
from upload import save_file
import uvicorn

ALLOWED_CONTENT_TYPES = {
    "image/jpg",
    "image/jpeg",
    "image/png",
}

app = FastAPI()


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
        for file in files:
            await save_file(file)
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"message": "Files uploaded successfully"}


if __name__ == "__main__":
    uvicorn.run(app)
