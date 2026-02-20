import random

import streamlit as st

from PIL import Image

uploaded_files = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files="directory",
    key="file_uploader",
)

if uploaded_files:
    st.write(f"Total images uploaded: {len(uploaded_files)}")

    sample_files = random.sample(uploaded_files, min(5, len(uploaded_files)))

    st.subheader("Sample of Uploaded Images")
    cols = st.columns(5)

    for col, file in zip(cols, sample_files):
        image = Image.open(file)
        col.image(image, caption=file.name, width="stretch")