import os
import random

import requests
import streamlit as st

from PIL import Image

uploaded_files = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files="directory",
    key="file_uploader",
)

if st.button("Submit"):
    if uploaded_files is not None:
        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        response = requests.post("http://localhost:8000/upload", files=files)
        st.write(response.text)
    else:
        st.write("No file uploaded.")
        
if st.button("View Uploaded Images"):
    response = requests.get("http://localhost:8000/images")
    if response.status_code == 200:
        data = response.json()
        images = data["images"]
        st.write(f"Total images: {data['total']}")
        if images:
            cols = st.columns(5)
            for col, filename in zip(cols, images[:5]):
                image = Image.open(os.path.join("uploads", filename))
                col.image(image, caption=filename)
        else:
            st.write("No images found.")
    else:
        st.write("Failed to retrieve images.")