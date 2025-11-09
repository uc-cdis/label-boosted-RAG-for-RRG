FROM quay.io/cdis/jupyter-superslim:master

LABEL name="jupyterlab-gpu-multiarch"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ----------------------------------------------------------
# Detect architecture and configure NVIDIA repo for AL2023
# ----------------------------------------------------------
ENV DISTRO="amzn2023"

RUN set -eux && \
    dnf -y update && \
    dnf install -y dnf-plugins-core && \
    arch=$(uname -m) && \
    if [[ "$arch" == "aarch64" ]]; then \
        # ARM64 (Graviton or A100G instances)
        dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/sbsa/cuda-${DISTRO}.repo; \
    else \
        # x86_64 (Intel/AMD GPU instances)
        dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-${DISTRO}.repo; \
    fi && \
    dnf clean expire-cache && \
    # Enable open kernel driver stream (universal)
    dnf -y module enable nvidia-driver:open-dkms && \
    # Install CUDA toolkit and open NVIDIA drivers (for container toolkit)
    dnf -y install nvidia-open cuda-toolkit && \
    dnf clean all

# NVIDIA Container Toolkit runtime vars
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ----------------------------------------------------------
# GPU Python Stack: PyTorch, TensorFlow (multi-arch aware)
# ----------------------------------------------------------
# The base image defines NB_UID/NB_USER
USER ${NB_UID}

# # Note: conda-forge and PyTorch wheels support both archs natively.
# RUN conda install -y -c conda-forge \
#       pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia && \
#     conda install -y tensorflow && \
#     conda clean -afy

# Optional: build JupyterLab extensions
# RUN jupyter lab build

# ----------------------------------------------------------
# Runtime configuration
# ----------------------------------------------------------
WORKDIR /home/${NB_USER}
EXPOSE 8888

# Copy notebooks
COPY demos/* .

# need to figure out why platform is aarch64 during image build pipeline
# RUN pip install h5py pandas numpy tqdm pqdm scikit-learn scikit-survival==0.23.1 jsonrpcclient gen3 hi-ml-multimodal==0.2.2 torch transformers accelerate openai

ENTRYPOINT ["/tini", "-g", "--"]
CMD ["start-notebook.sh"]
