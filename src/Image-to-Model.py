import torch
import numpy as np
from diffusers import DiffusionPipeline
from PIL import Image

img_nobg_jpg_path = "src/tmp/image_nobg.jpg"
model_ply_path = "src/tmp/model.ply"


pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM-full",
    custom_pipeline="dylanebert/LGM-full",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

input_image = Image.open(img_nobg_jpg_path)
input_image = np.array(input_image, dtype=np.float32) / 255.0
result = pipeline("", input_image)
pipeline.save_ply(result, model_ply_path)

