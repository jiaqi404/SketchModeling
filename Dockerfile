FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN mkdir -p /workspace/sketchmodeling
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace/sketchmodeling
ADD ./requirements.txt /workspace/sketchmodeling/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace/sketchmodeling

# set CUDA_HOME
ENV CUDA_HOME="/usr/local/cuda-12.4:$CUDA_HOME"

# clear cache
RUN pip cache purge

CMD ["python", "app.py"]