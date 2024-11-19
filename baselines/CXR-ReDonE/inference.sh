source ~/.bashrc

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/11.2

cd ALBEF

python3 CXR_ReDonE_pipeline.py 
