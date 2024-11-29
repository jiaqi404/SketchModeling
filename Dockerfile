FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN mkdir -p /workspace/sketchmodeling
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/workspace/miniconda3/bin:${PATH}"

RUN conda init bash

RUN conda create -n sketchmodeling python=3.12 && echo "source activate sketchmodeling" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/instantmesh/bin:$PATH

RUN conda install Ninja
RUN conda install cuda -c nvidia/label/cuda-12.4.1 -y
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace/sketchmodeling
ADD ./requirements.txt /workspace/sketchmodeling/requirements.txt
RUN pip install -r requirements.txt

COPY . /workspace/sketchmodeling

# clear cache
RUN pip cache purge

CMD ["python", "app.py"]