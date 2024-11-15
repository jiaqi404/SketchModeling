from transformers import pipeline

img_path = "src/tmp/image.png"
img_nobg_png_path = "src/tmp/image_nobg.png"
img_nobg_path = "src/tmp/image_nobg.jpg"


pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=0)

image_nobg = pipe(img_path)
image_nobg.save(img_nobg_path)
