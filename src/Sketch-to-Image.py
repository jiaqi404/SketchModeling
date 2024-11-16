from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image

sketch_path = "src/tmp/sketch.png"
img_path = "src/tmp/image.png"
prompt = "sneaker"
add_prompt = ", realistic rendering, high contrast, white background"
prompt += add_prompt
negative_prompt = "black and white image"

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

sketch_image=Image.open(sketch_path)
output = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    controlnet_conditioning_scale=0.8, 
    image=sketch_image
).images[0]

output.save(img_path)