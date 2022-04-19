#!/bin/sh
#SBATCH --job-name=low-lr-6-epoch-jobber
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stevenjust4edu@gmail.com
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=run_%j.out #append job id
#SBATCH -D /gpfs/u/home/MLI2/MLI2wngk/scratch/MLBinfCapstone/

python evoformer_trainer.py