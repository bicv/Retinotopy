#!/bin/sh
#SBATCH -J Retino
#SBATCH  -A fsx@v100
#SBATCH  -C v100-32g
## http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-exec_mono_batch.html
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -t 40:00:00 
#SBATCH -o cached_data/log_%j_out.log  # <-- the name of the file where the output of the simulation is written
#SBATCH -e cached_data/log_%j_err.log  # <-- the name of the file where errors of the simulation are written


module purge
module load pytorch-gpu/py3
make
