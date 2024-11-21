# get the development image from nvidia cuda 12.4
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL name="sketchmodeling" maintainer="sketchmodeling"

# Add a volume for downloaded models
VOLUME /workspace/models

# create workspace folder and set it as working directory
RUN mkdir -p /workspace/sketchmodeling
WORKDIR /workspace

# update package lists and install git, wget, vim, libegl1-mesa-dev, and libglib2.0-0
RUN apt-get update && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

# create and activate conda environment
RUN conda create -n sketchmodeling python=3.12 && echo "source activate sketchmodeling" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/sketchmodeling/bin:$PATH

# install pytorch & xformers
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu124

# change the working directory to the repository
WORKDIR /workspace/sketchmodeling

# install other dependencies
ADD ./requirements.txt /workspace/sketchmodeling/requirements.txt
RUN pip install -r requirements.txt

COPY . /workspace/sketchmodeling

# Run the command when the container starts
CMD ["python", "app.py"]