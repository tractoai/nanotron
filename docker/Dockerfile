FROM cr.ai.nebius.cloud/crnf2coti090683j5ssi/kernels/default:2024-09-23-17-29-46-81de2bdf

USER root
RUN python3 -m pip install ytsaurus-client ytsaurus-yson
RUN python3 -m pip install ytsaurus-rpc-driver
RUN git clone https://github.com/Pleias/nanotron-tractoai.git
RUN pip install --upgrade pip
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip install -e ./nanotron-tractoai

# Install dependencies if you want to use the example scripts
RUN pip install datasets transformers

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-4

RUN pip install packaging ninja

RUN pip install triton "flash-attn>=2.5.0" --no-build-isolation

RUN pip install "tractorun==0.45.0"
RUN pip install "datatrove[io]"
RUN pip install numba
COPY gai.conf /etc/gai.conf

ENV PATH /usr/local/cuda-12.4/bin:$PATH
ENV CUDA_HOME /usr/local/cuda-12.4

USER 1000
ENV PATH /usr/local/cuda-12.4/bin:$PATH
ENV CUDA_HOME /usr/local/cuda-12.4
