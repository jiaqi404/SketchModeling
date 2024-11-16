import numpy as np
import torch
from diffusers import DiffusionPipeline
from torchvision.transforms import v2
from einops import rearrange
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)

# config_path = 'configs/instant-mesh-large.yaml'
# config = OmegaConf.load(config_path)
# config_name = os.path.basename(config_path).replace('.yaml', '')
# model_config = config.model_config
# infer_config = config.infer_config

# IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

# device = torch.device('cuda')

# load diffusion model
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16
).to('cuda')
# pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
#     pipeline.scheduler.config, timestep_spacing='trailing'
# )

# load custom white-background UNet
# unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model", cache_dir=model_cache_dir)
# state_dict = torch.load(unet_ckpt_path, map_location='cpu')
# pipeline.unet.load_state_dict(state_dict, strict=True)


# load reconstruction model
# model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model", cache_dir=model_cache_dir)
# model = instantiate_from_config(model_config)
# state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
# state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
# model.load_state_dict(state_dict, strict=True)

# model = model.to(device1)
# if IS_FLEXICUBES:
#     model.init_flexicubes_geometry(device1, fovy=30.0)
# model = model.eval()

# print('Loading Finished!')


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

def image_to_model(input_img):
    input_img = np.asarray(input_img, dtype=np.float32) / 255.0
    input_img = torch.from_numpy(input_img ).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    input_img  = rearrange(input_img, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    device = torch.device('cuda')
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device)

    input_img = input_img.unsqueeze(0).to(device)
    input_img = v2.functional.resize(input_img, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    model_path = "src/tmp/model.obj"
    model_glb_path = "src/tmp/model.glb"

    # mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    # print(mesh_fpath)
    # mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    # mesh_dirname = os.path.dirname(mesh_fpath)
    # video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(input_img, input_cameras)

        # get video
        # chunk_size = 20 if IS_FLEXICUBES else 1
        # render_size = 384
        
        # frames = []
        # for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        #     if IS_FLEXICUBES:
        #         frame = model.forward_geometry(
        #             planes,
        #             render_cameras[:, i:i+chunk_size],
        #             render_size=render_size,
        #         )['img']
        #     else:
        #         frame = model.synthesizer(
        #             planes,
        #             cameras=render_cameras[:, i:i+chunk_size],
        #             render_size=render_size,
        #         )['images_rgb']
        #     frames.append(frame)
        # frames = torch.cat(frames, dim=1)

        # images_to_video(
        #     frames[0],
        #     video_fpath,
        #     fps=30,
        # )

        # print(f"Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

    return mesh_fpath, mesh_glb_fpath