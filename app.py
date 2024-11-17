import gradio as gr
import os
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from torchvision.transforms import v2
from einops import rearrange
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from src.utils.mesh_util import save_obj, save_glb
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)

# ----------------------Settings-----------------------

# Define the cache directory for model files
model_cache_dir = 'ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

# Configuration
config_path = 'configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

# Device
device = torch.device('cuda')

# ----------------------Load Models-----------------------

# load models for sketch-to-image
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble", 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    cache_dir=model_cache_dir
)
pipeline_1 = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    cache_dir=model_cache_dir
)
pipeline_1.scheduler = UniPCMultistepScheduler.from_config(pipeline_1.scheduler.config)
pipeline_1.enable_model_cpu_offload()

# load models for background remove
pipeline_2 = pipeline(
    "image-segmentation", 
    model="briaai/RMBG-1.4", 
    trust_remote_code=True, 
    device=0
)
pipeline_2.save_pretrained(model_cache_dir)

# load models for image-to-model
pipeline_3 = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
    cache_dir=model_cache_dir
)
pipeline_3.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline_3.scheduler.config, timestep_spacing='trailing'
)
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model", cache_dir=model_cache_dir)
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline_3.unet.load_state_dict(state_dict, strict=True)
pipeline_3 = pipeline_3.to(device)

# load models for 3d model reconstruction
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model", cache_dir=model_cache_dir)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

print('----------------------Loading Finished-----------------------')

# ----------------------Define functions-----------------------

def input_image(input_img):
    if input_img is None:
        raise gr.Error("No image uploaded!")
    else:
        input_img.save("src/tmp/sketch.png")
        return

def sketch_to_image(
        input_img, 
        prompt, 
        negative_prompt="low quality, black and white image", 
        add_prompt=", 3d rendered, shadeless, white background, intact and single object", 
        controlnet_conditioning_scale=0.75,
        num_inference_steps=50
    ):

    output = pipeline_1(
        prompt+add_prompt, 
        num_inference_steps=int(num_inference_steps),
        guidance_scale=10,
        negative_prompt=negative_prompt, 
        controlnet_conditioning_scale=float(controlnet_conditioning_scale), 
        image=input_img
    ).images[0]

    output.save("src/tmp/image.png")

    return output

def background_remove(input_img):
    output = pipeline_2(input_img)
    output.save("src/tmp/image_nobg.png")
    
    return output

def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

def make_mesh(model_path, model_glb_path, planes):
    with torch.no_grad():
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        
        save_obj(vertices, faces, vertex_colors, model_path)
        save_glb(vertices, faces, vertex_colors, model_glb_path)

    return model_path, model_glb_path

def image_to_model(input_img):
    generator = torch.Generator(device=device)
    z123_image = pipeline_3(
        input_img,
        generator=generator,
    ).images[0]

    input_img = np.asarray(z123_image, dtype=np.float32) / 255.0
    input_img = torch.from_numpy(input_img ).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    input_img  = rearrange(input_img, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)

    input_img = input_img.unsqueeze(0).to(device)
    input_img = v2.functional.resize(input_img, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    model_path = "src/tmp/model.obj"
    model_glb_path = "src/tmp/model.glb"

    with torch.no_grad():
        planes = model.forward_planes(input_img, input_cameras)

    model_path, model_glb_path = make_mesh(model_path, model_glb_path, planes)

    return model_path, model_glb_path

# ----------------------Build Gradio Interfaces-----------------------

with gr.Blocks() as demo:
    gr.Markdown("""
        # SketchModeling: From Sketch to 3D Model

        **SketchModeling** is a method for 3D mesh generation from a sketch.

        It has three steps:
        1. It generates image from sketch using stable diffusion and controlnet.
        2. It removes the background of the image using RMBG.
        3. It reconsturcted the 3D model of the image using LGM.

        On below, you can either upload a sketch image or draw the sketch yourself. Then press Run and wait for the model to be generated.
        """)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_img = gr.Image(
                    type="pil", label="Input Image", sources="upload", image_mode="RGBA"
                )
                generated_img = gr.Image(
                    type="pil", label="Gnerated Image", image_mode="RGBA", interactive=False
                )
                processed_img = gr.Image(
                    type="pil", label="Processed Image", image_mode="RGBA", interactive=False
                )
            with gr.Row():
                prompt = gr.Textbox(label="Pompt", interactive=True)
                controlnet_conditioning_scale = gr.Slider(
                    label="Controlnet Conditioning Scale",
                    minimum=0.5,
                    maximum=1.5,
                    value=0.85,
                    step=0.05,
                    interactive=True
                )
            with gr.Accordion('Advanced options', open=False):
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, black and white image", interactive=True)
                    add_prompt = gr.Textbox(label="Styles", value=", 3d rendered, shadeless, white background, intact and single object", interactive=True)
                    num_inference_steps = gr.Number(label="Inference Steps", value=50, interactive=True)
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column():
            with gr.Tab("OBJ"):
                output_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    interactive=False
                )
            with gr.Tab("GLB"):
                output_glb = gr.Model3D(
                    label="Output Model (GLB Format)",
                    interactive=False
                )

    run_btn.click(fn=input_image, inputs=[input_img]).success(
        fn=sketch_to_image,
        inputs=[input_img, prompt, negative_prompt, add_prompt, controlnet_conditioning_scale, num_inference_steps],
        outputs=[generated_img]
    ).success(
        fn=background_remove,
        inputs=[generated_img],
        outputs=[processed_img]
    ).success(
        fn=image_to_model,
        inputs=[processed_img],
        outputs=[output_obj, output_glb]
    )

demo.launch()