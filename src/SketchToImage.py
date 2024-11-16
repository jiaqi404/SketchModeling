from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

def sketch_to_image(
        input_img, 
        prompt, 
        negative_prompt="low quality, black and white image", 
        add_prompt=", 3d rendered, shadeless, white background, intact and single object", 
        controlnet_conditioning_scale=0.75,
        num_inference_steps=50
    ):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt+add_prompt, 
        num_inference_steps=int(num_inference_steps),
        guidance_scale=10,
        negative_prompt=negative_prompt, 
        controlnet_conditioning_scale=float(controlnet_conditioning_scale), 
        image=input_img
    ).images[0]

    output.save("src/tmp/image.png")

    return output