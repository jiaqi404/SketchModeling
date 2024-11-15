import torch
import requests
import numpy as np
from io import BytesIO
from diffusers import DiffusionPipeline
from PIL import Image

pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM-full",
    custom_pipeline="dylanebert/LGM-full",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

input_url = "https://huggingface.co/datasets/dylanebert/iso3d/resolve/main/jpg@512/a_cat_statue.jpg"
input_image = Image.open(BytesIO(requests.get(input_url).content))
input_image = np.array(input_image, dtype=np.float32) / 255.0
result = pipeline("", input_image)
result_path = "/tmp/output.ply"
pipeline.save_ply(result, result_path)
