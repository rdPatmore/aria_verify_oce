#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --mem=50GB
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --account=verify_oce
source activate verify

python -u plot_ensemble.py
