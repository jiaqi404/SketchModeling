# SketchModeling: From Sketch to 3D Model
**SketchModeling** proposes a new method combining painting frame and natural language processing, through three steps of image synthesis, background removal and 3D modeling, with multi-modal input, automatic background removal and integrated modeling process as innovation points, which has important technical value and application prospect. An innovative approach that increases modeling efficiency and provides new ways to create across multiple domains.

## Get Started
This is an example of how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

### Prerequisites


### Install Dependencies
- Recommend using: python=3.12 + cuda=12.4
- Install CUDA of your computer’s support version. In my case, I downloaded CUDA 12.4 through this [link](https://developer.nvidia.com/cuda-12-4-0-download-archive) (**NOT THROUGH CONDA**)
- if you have problem installing CUDA, try installing Nsight separatly, then installing CUDA in advanced mode and uncheck Nsight.
- Install the right version of [pytorch](https://pytorch.org/) and [xformers](https://github.com/facebookresearch/xformers)
- Install the remaining requirements with `pip install -r requirements.txt`

## Usage
![Alt text](images/screenshot.png)
This is our demo video:https://youtu.be/BoggiFAqmmY

### Roadmap
![Alt text](images/roadmap.png)

## Contributing

### Top contributors:

<div style="display: flex; justify-content: space-between;">
  <img src="images/zjq1.png" alt="ZJQ Image" style="width: 10%;"/>
  <img src="images/zsn1.png" alt="ZSN Image" style="width: 10%;"/>
  <img src="images/zyh1.png" alt="ZYH Image" style="width: 10%;"/>
</div>

## License
Distributed under the MIT License. See LICENSE.txt for more information.

## Contact
Zhang Jiaqi marycheung021213@gmail.com
Zhou Shengnan 24058989g@connect.polyu.hk
Zeng Yihan zengyihan9@gmail.com

## Acknowledgments

### Load Gradio App
```sh
python app.py
```
