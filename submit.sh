#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=download_data
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --account=engs-pnpl
#SBATCH --output=results/slurm_out/%j.out

module load Anaconda3/2022.10
conda activate /data/engs-pnpl/trin4076/arc_pytorch
export WANDB_CACHE_DIR=$DATA/wandb_cache

pip install -r requirements.txt
python main.py
