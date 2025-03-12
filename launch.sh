#!/bin/sh
#SBATCH -J Retino
#SBATCH  -A fsx@v100
#SBATCH  -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -t 24:00:00 
#SBATCH -o cached_data/log_%j_out.log  # <-- the name of the file where the output of the simulation is written
#SBATCH -e cached_data/log_%j_err.log  # <-- the name of the file where errors of the simulation are written
# https://mesocentre.univ-amu.fr/slurm/

module purge
module load pytorch-gpu/py3

make
