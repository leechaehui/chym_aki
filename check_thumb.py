from PIL import Image
import os
tp = 'D:/chym_aki_data/metadata/thumbnails/30-10034_HE_ae019f79_thumb.jpg'
if os.path.exists(tp):
    img = Image.open(tp)
    print("Thumbnail size:", img.size)
else:
    print("Thumbnail not found!")
