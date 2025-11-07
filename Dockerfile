FROM quay.io/cdis/jupyter-superslim:master 

# (continue from your Dockerfile end)
# GPU-ready JupyterLab variant

# Switch to root to install CUDA and dependencies
USER root

# Install CUDA toolkit and cuDNN runtime matching host drivers
RUN yum install -y \
      nvidia-driver-latest-dkms \
      cuda-toolkit-12-4 \
      libcudnn9-cuda12 && \
    yum clean all

# Environment for NVIDIA Container Toolkit runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install GPU-accelerated Python packages
USER ${NB_UID}
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia && \
    conda install -y tensorflow && \
    conda clean -afy

# Optional: ensure JupyterLab extensions work fine
RUN jupyter lab build

EXPOSE 8888

COPY . .

ENTRYPOINT ["/tini", "-g", "--"]
CMD ["start-notebook.sh"]
