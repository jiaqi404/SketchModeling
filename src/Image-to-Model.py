import torch
import numpy as np
from diffusers import DiffusionPipeline
from PIL import Image

img_nobg_path = "src/tmp/image_nobg.png"

pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM-full",
    custom_pipeline="dylanebert/LGM-full",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

input_image = Image.open(img_nobg_path)
input_image = np.array(input_image, dtype=np.float32) / 255.0
result = pipeline("", input_image)
result_path = "src/tmp/output.ply"
pipeline.save_ply(result, result_path)
