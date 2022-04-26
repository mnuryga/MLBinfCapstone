#!/bin/sh
#SBATCH --job-name=jobbidy-jobber-jobber-job-job-job-jobbidy-job-job-job-joblib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stevenjust4edu@gmail.com
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=jobber_%j.out #append job id
#SBATCH -D /gpfs/u/home/MLI2/MLI2wngk/scratch/MLBinfCapstone/

python trainer.py