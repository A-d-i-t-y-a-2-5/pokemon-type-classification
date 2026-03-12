from concurrent.futures import ThreadPoolExecutor
import os
import random

import requests
import streamlit as st

from PIL import Image

MAX_BATCH_SIZE = 1000

uploaded_files = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files="directory",
    key="file_uploader",
)

if st.button("Submit"):
    if uploaded_files is not None:
        total = len(uploaded_files)
        batches = [
            uploaded_files[i : i + MAX_BATCH_SIZE]
            for i in range(0, total, MAX_BATCH_SIZE)
        ]

        def upload_batch(batch):
            files = [("files", (file.name, file, file.type)) for file in batch]
            return requests.post("http://fastapi:8000/upload", files=files)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(upload_batch, batch) for batch in batches]
            for future in futures:
                response = future.result()
                st.write(response.text)

        process_response = requests.post("http://fastapi:8000/process")
        if process_response.status_code == 200:
            st.success(process_response.text)
        else:
            st.error(f"Vectorization trigger failed: {process_response.text}")
    else:
        st.write("No file uploaded.")

if st.button("View Uploaded Images"):
    response = requests.get("http://fastapi:8000/images")
    if response.status_code == 200:
        data = response.json()
        images = data["images"]
        st.write(f"Total images: {data['total']}")
        if images:
            cols = st.columns(5)
            for col, filename in zip(cols, images[:5]):
                image = Image.open(os.path.join("/app/uploads", filename))
                col.image(image, caption=filename)
        else:
            st.write("No images found.")
    else:
        st.write("Failed to retrieve images.")

st.subheader("Search Similar Images")
query = st.text_input("Enter a search query", placeholder="e.g. a dog on a beach")
if st.button("Search"):
    if query:
        response = requests.post(
            "http://fastapi:8000/search",
            json={"query": query, "top_k": 5},
        )
        if response.status_code == 200:
            data = response.json()
            results = data["results"]
            # # st.write(f"Top {len(results)} similar images")
            # st.write(results)
            if results:
                cols = st.columns(5)
                for col, result in zip(cols, results):
                    image = Image.open(os.path.join("/app/uploads", result))
                    col.image(image, caption=result, width="stretch")
            else:
                st.write("No similar images found.")
        else:
            st.write(f"Search failed due to {response.text}")
    else:
        st.write("Please enter a query.")
