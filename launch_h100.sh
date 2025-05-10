#!/bin/sh
# http://www.idris.fr/eng/jean-zay/modifications-extension-jean-zay-h100-eng.html
#SBATCH -J Retino
#SBATCH  -A fsx@h100
#SBATCH  -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -t 19:59:00
#SBATCH -o cached_data/log_%j_out.log  # <-- the name of the file where the output of the simulation is written
#SBATCH -e cached_data/log_%j_err.log  # <-- the name of the file where errors of the simulation are written

module purge
make module load arch/h100
module load pytorch-gpu/py3

make
