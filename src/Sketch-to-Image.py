from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image

sketch_path = "src/tmp/sketch.png"
img_path = "src/tmp/image.png"
prompt = "hot air balloon"
add_prompt = ", 3d rendered, shadowless, shadeless, white background, intact and single object"
prompt += add_prompt
negative_prompt = "low quality, black and white image"

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

sketch_image=Image.open(sketch_path)
output = pipe(
    prompt, 
    num_inference_steps=200,
    guidance_scale=10,
    negative_prompt=negative_prompt, 
    controlnet_conditioning_scale=0.75, 
    image=sketch_image
).images[0]

output.save(img_path)