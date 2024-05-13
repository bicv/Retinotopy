DIR=JeremieDaucePerrinet2024notebooks

#################@#################@#################@#################
# bootstrap: rsync  -av -u --exclude *.pt  -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa" /data/perrinet.l/research/science/JNJER_PhD/RetinoTopy/JeremieDaucePerrinet2024notebooks lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science
# data rsync  -av -u --exclude *.pt  -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa" /data/JNJER/Deep_learning/data/Imagenet_focus lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/Deep_learning/data/
# rsync  -av -u --exclude *.pt  -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa" /data/JNJER/Deep_learning/data/Imagenet_bbox lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/Deep_learning/data/

# 
MESO_URL = lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/
MESO_OPTS=-av -u --exclude *.pt --exclude pytorch.sif   -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa"
pull_meso:
	rsync $(MESO_OPTS)   $(MESO_URL)/$(DIR)/cached_data .

push_meso:
	rsync  $(MESO_OPTS) cached_data $(MESO_URL)/$(DIR) 

pull_all: pull_ada pull_babbage pull_darwin pull_fortytwo pull_meso
push_all: push_ada push_babbage push_darwin push_fortytwo push_meso

#################@#################@#################@#################

load_modules:
	module load userspace/all; module load cuda/10.1

# https://arcca.github.io/intro_singularity/06-singularity-gpu/index.html
singularity_build:
	singularity build pytorch.sif pytorch.def

singularity:
	singularity shell --bind /scratch:/scratch --nv pytorch.sif

#################@#################@#################@#################

update:
	python3 -m pip install --upgrade -r requirements.txt
