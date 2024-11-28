# SketchModeling: From Sketch to 3D Model
**SketchModeling** is a method for 3D mesh generation from a sketch, through three steps of image generation, background removal and 3D reconstruction.
![Alt text](assets/outputs-result.png)

### How it works
![Alt text](assets/model-structrue.png)

## Getting Started
### Install Dependencies
- Recommend using: python=3.12 + cuda=12.4
- Install CUDA of your computer's support version. In my case, I downloaded CUDA 12.4 through this [link](https://developer.nvidia.com/cuda-12-4-0-download-archive) (**NOT THROUGH CONDA**)
- if you have problem installing CUDA, try installing Nsight separatly, then installing CUDA in advanced mode and uncheck Nsight
- Install the right version of [pytorch](https://pytorch.org/) and [xformers](https://github.com/facebookresearch/xformers)
- Install the remaining requirements with `pip install -r requirements.txt`
```sh
# Create conda env
conda create --name sketchmodeling python=3.12
conda activate sketchmodeling

# Install PyTorch and xformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install -r requirements.txt
```

### Load Gradio App
```sh
python app.py
```

### Demo Video
link: https://youtu.be/BoggiFAqmmY

### Start with Docker
*⚠️WARNING⚠️: It is under testing. Do not use it at present. --2024/11/26*

You can also use Docker to set up environment automatically.
- Open [Docker desktop](https://www.docker.com/products/docker-desktop/)
- Build docker image with `docker build -t sketchmodeling .`
- Run docker image with `docker run -it --platform=linux/amd64 --gpus all sketchmodeling`

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
- Zhang Jiaqi marycheung021213@gmail.com
- Zhou Shengnan 24058989g@connect.polyu.hk
- Zeng Yihan zengyihan9@gmail.com

## Acknowledgments
We thank the authors of the following projects for their excellent contributions!
- [ControlNet](https://github.com/lllyasviel/ControlNet?tab=readme-ov-file)
- [RMGB](https://github.com/ai-anchorite/BRIA-RMBG-2.0)
- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [InstantMesh](https://github.com/TencentARC/InstantMesh)
