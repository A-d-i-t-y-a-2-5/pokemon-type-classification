from PIL import Image
from glob import glob

def delete_iccfile(image_path: str):
    img = Image.open(image_path)
    img.info.pop('icc_profile', None)
    img.save(image_path)
    
img_paths = glob("./data/images/*/*/*.png")
for img_path in img_paths:
    delete_iccfile(img_path)