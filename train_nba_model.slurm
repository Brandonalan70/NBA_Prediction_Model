#!/bin/bash
#SBATCH --job-name=nba_logistic_regression
#SBATCH --output=slurm-nba-model-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=RM
#SBATCH --account=cis240116p
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bfb5559@psu.edu

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

module load anaconda3

conda create -y -n nba_env python=3.9
source activate nba_env

pip install pandas numpy scikit-learn joblib

cd $PROJECT/nba_modeling

echo "Starting model training..."
python train_nba_logistic.py

echo "Job completed at: $(date)"