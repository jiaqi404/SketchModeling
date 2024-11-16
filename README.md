# SketchModeling: From Sketch to 3D Model
**SketchModeling** is a method for 3D mesh reconstruction from a sketch.

## Get Started
### Installation tips
*my environment: python 3.12.4 + cuda 12.4 + win-64
- Make sure you have installed CUDA of your computer’s support version. In my case, I downloaded CUDA 12.4 through this link[https://developer.nvidia.com/cuda-12-4-0-download-archive] below (NOT THROUGH CONDA)
- if you have problem installing CUDA, try installing Nsight separatly, then installing CUDA in advanced mode and uncheck Nsight.
- Make sure you have set a new system environment variable named CUDA_HOME, which path in my case is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
- Make sure you have installed the right version of pytorch[https://pytorch.org/] and xformers[https://github.com/facebookresearch/xformers]
- Install the remaining requirements with `pip install -r requirements.txt`

### Load Gradio App
```sh
python app.py
```


