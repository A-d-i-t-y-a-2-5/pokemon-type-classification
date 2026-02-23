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