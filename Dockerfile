FROM quay.io/cdis/jupyter-superslim:master

LABEL name="jupyterlab-gpu-multiarch"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install linux dependencies for building python wheels
RUN yum install -y wget python3-devel gcc gcc-c++ && yum clean all && rm -rf /var/cache/yum

USER ${NB_UID}
WORKDIR /home/${NB_USER}
EXPOSE 8888

# Copy demo materials
COPY demos/*.ipynb demos/requirements.txt .
COPY demos/manifests/ ./manifests/

# Install python dependencies
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install -r requirements.txt \
 && pip cache purge

ENTRYPOINT ["/tini", "-g", "--"]
CMD ["start-notebook.sh"]
