FROM quay.io/cdis/jupyter-superslim:master

LABEL name="jupyterlab-gpu-multiarch"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN yum install -y wget python3-devel gcc gcc-c++ && yum clean all && rm -rf /var/cache/yum

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

# Copy demo materials
COPY demos/*.ipynb demos/manifests requirements-docker.txt .

RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install -r requirements-docker.txt \
 && pip cache purge

ENTRYPOINT ["/tini", "-g", "--"]
CMD ["start-notebook.sh"]
