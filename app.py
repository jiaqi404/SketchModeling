import gradio as gr
from gradio_litmodel3d import LitModel3D
import os
from src.SketchToImage import sketch_to_image
from src.BackgroundRemove import background_remove
import numpy as np
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from src.utils.mesh_util import save_obj
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
model_cache_dir = 'ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

seed_everything(0)

config_path = 'configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
    cache_dir=model_cache_dir
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model", cache_dir=model_cache_dir)
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device0)

# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model", cache_dir=model_cache_dir)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()

print('Loading Finished!')

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

def make_mesh(model_path, planes):
        
    with torch.no_grad():
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        
        save_obj(vertices, faces, vertex_colors, model_path)

    return model_path

def image_to_model(input_img):
    # sampling
    generator = torch.Generator(device=device0)
    z123_image = pipeline(
        input_img,
        generator=generator,
    ).images[0]

    input_img = np.asarray(z123_image, dtype=np.float32) / 255.0
    input_img = torch.from_numpy(input_img ).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    input_img  = rearrange(input_img, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    device = torch.device('cuda')
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    # render_cameras = get_render_cameras(
    #     batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device)

    input_img = input_img.unsqueeze(0).to(device)
    input_img = v2.functional.resize(input_img, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    model_path = "src/tmp/model.obj"

    with torch.no_grad():
        planes = model.forward_planes(input_img, input_cameras)

    model_path = make_mesh(model_path, planes)

    return model_path

def input_image(input_img):
    input_img.save("src/tmp/sketch.png")
    return

with gr.Blocks() as demo:
    gr.Markdown("""
        # SketchModeling: From Sketch to 3D Model

        **SketchModeling** is a method for 3D mesh reconstruction from a sketch.

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
            output_3d = LitModel3D(
                label="3D Model",
                visible=True,
                clear_color=[0.0, 0.0, 0.0, 0.0],
                tonemapping="aces",
                contrast=1.0,
                scale=1.0,
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
        outputs=[output_3d]
    )

demo.launch()