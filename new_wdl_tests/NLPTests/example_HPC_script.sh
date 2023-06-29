#!/bin/bash
#SBATCH -n 8
#SBATCH --time=02-00:00:00
#SBATCH -p gpu
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --array=1-30
#SBATCH --mail-type=BEGIN,END,FAIL


source /path/to/.bashrc
conda activate wdl
cd /path/to/ICMLCode/new_wdl_tests/NLPTests/
python BCM_extension.py --trial $SLURM_ARRAY_TASK_ID --locality $1


# a meta bash script was used to launch this script:
#for locality in 10.0 1.0 0.1 0.01 0.0
#do
#	sbatch bcm_comparison_job.sh $locality
#done
