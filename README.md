# SketchModeling: From Sketch to 3D Model
**SketchModeling** is a method for 3D mesh generation from a sketch, through three steps of image generation, background removal and 3D modeling.

## About SketchModeling
### How it works
![Alt text](images/model-structure.png)

## Getting Started
### Install Dependencies
- Recommend using: python=3.12 + cuda=12.4
- Install CUDA of your computer’s support version. In my case, I downloaded CUDA 12.4 through this [link](https://developer.nvidia.com/cuda-12-4-0-download-archive) (**NOT THROUGH CONDA**)
- if you have problem installing CUDA, try installing Nsight separatly, then installing CUDA in advanced mode and uncheck Nsight.
- Install the right version of [pytorch](https://pytorch.org/) and [xformers](https://github.com/facebookresearch/xformers)
- Install the remaining requirements with `pip install -r requirements.txt`
```sh
conda create --name sketchmodeling python=3.12
conda activate sketchmodeling

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different python & cuda version
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
You can also use Docker to set up environment. This docker setup is tested on Ubuntu.
- Build docker image with `docker build -t sketchmodeling .`
- Run docker image with a local model cache (so it is fast when container is started next time):
```sh
mkdir -p $HOME/models/
export MODEL_DIR=$HOME/models/

docker run -it --platform=linux/amd64 --gpus all -v $MODEL_DIR:/workspace/sketchmodeling/models sketchmodeling
```

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

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
