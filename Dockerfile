FROM quay.io/cdis/jupyter-superslim:master

LABEL name="jupyterlab-gpu-multiarch"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Circumvent SSL issues by predownload data files (for full-data demos)
# Can remove this when batched embedding download enables faster data loading
ADD "https://uchicago.box.com/shared/static/er0sz3yt17ke9zyg97ruhgemntuhx4r1.h5" ./with-all-data/data/openi.h5
ADD "https://uchicago.box.com/shared/static/rasdb3b3xuirx7q4k012vnnx558px3we.csv" ./with-all-data/data/openi.csv
ADD "https://uchicago.box.com/shared/static/k8z0kip2pej2v62pwgymdm45gt7dq0yw.h5" ./with-all-data/data/hist.h5
ADD "https://uchicago.box.com/shared/static/hr82b5c9g3h4y8c7avrnbgvhgdcoetld.h5" ./with-all-data/data/expr.h5
ADD "https://uchicago.box.com/shared/static/liwt3vlvdpmbfsa21wqboshh9nv6enm2.h5" ./with-all-data/data/summ.h5
RUN chmod -R a+rwx ./with-all-data

# Install linux dependencies for building python wheels
RUN yum install -y python3-devel gcc gcc-c++ && yum clean all && rm -rf /var/cache/yum

USER ${NB_UID}
WORKDIR /home/${NB_USER}
EXPOSE 8888

# Copy demo materials
COPY demos/ ./

# Install python dependencies
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install -r requirements.txt \
 && pip cache purge

# Circumvent SSL issues by predownloading model weights (for labrag-demos)
RUN hf download "ibm-granite/granite-4.0-h-350M" \
 && hf download --revision v1.0 "microsoft/BiomedVLP-BioViL-T" "biovil_t_image_model_proj_size_128.pt"

ENTRYPOINT ["/tini", "-g", "--"]
CMD ["start-notebook.sh"]
