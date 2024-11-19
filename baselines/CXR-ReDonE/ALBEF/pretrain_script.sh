source ~/.bashrc

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/11.2
nvidia-smi

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env Pretrain.py --config configs/Pretrain.yaml --output_dir output/pretrain/  --checkpoint ALBEF_4M.pth  --resume true
