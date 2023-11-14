#!/bin/bash
#SBATCH --job-name=WGAN
#SBATCH --output=WGAN.txt
#SBATCH --time=96:00:00
#SBATCH --nodes=1
###SBATCH --account=rwth0925
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Read user variables
source $HOME/miniconda3/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate WGAN

module load CUDA

# Print some debug information
echo; export; echo; nvidia-smi; echo

$CUDA_ROOT/extras/demo_suite/deviceQuery -noprompt

python run_cwgangp.py
