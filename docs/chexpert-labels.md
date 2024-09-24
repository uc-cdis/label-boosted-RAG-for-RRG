# CheXpert Labels
TODO incorporate into evaluation and explain purpose of labeling

## To generate chexpert labels
1. Install `docker` using the following instructions: https://docs.docker.com/engine/install/
    1. **NB:** *Please* don't install from the default apt channels, follow the instructions from docker
    1. Make sure to do the post-install steps too: https://docs.docker.com/engine/install/linux-postinstall/
    1. (Optional) If the drive that `/var/lib/docker` lives on is small, it might be a good idea to move the docker directory: https://www.ibm.com/docs/en/z-logdata-analytics/5.1.0?topic=software-relocating-docker-root-directory
1. Clone chexpert-labeler and build the docker image: https://github.com/stanfordmlgroup/chexpert-labeler?tab=readme-ov-file#dockerized-labeler
1. Run the chexper labeler docker dispatcher:
    ```
    python /path/to/rrg-repo/chexpert-labels/label-dispatcher.py \
    --n_jobs=$(nproc) \
    --data_path /path/to/rrg-data \
    --output_path /path/to/rrg-data/labels
    ```
1. You should end up with a `labels` folder with the `train/val` labels and the log files for each processed chunk. Please refer to the chexpert resources for a description of the labels.

## References
* TODO ref to chexpert