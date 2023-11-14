#!/bin/bash
#SBATCH --job-name=V3_KG
#SBATCH --output=V3_KG.txt
#SBATCH --time=96:00:00
#SBATCH --nodes=1
##SBATCH --account=thes1471
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=1

module load Python/3.10.4
python3 main.py
