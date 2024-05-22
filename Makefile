DIR=JeremieDaucePerrinet2024notebooks

#################@#################@#################@#################
# 
MESO_URL = lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/
MESO_OPTS=-av -u --exclude pytorch.sif   -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa"
pull_meso:
	rsync $(MESO_OPTS)   $(MESO_URL)/$(DIR)/cached_data .

push_meso:
	rsync  $(MESO_OPTS) cached_data $(MESO_URL)/$(DIR) 


# transfer dataset to the mesocentre
# bootstrap: rsync  -av -u --exclude *.pt  -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa" /data/perrinet.l/research/science/JNJER_PhD/RetinoTopy/JeremieDaucePerrinet2024notebooks lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science
# data rsync  -av -u --exclude *.pt  -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa" /data/JNJER/Deep_learning/data/Imagenet_focus lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/Deep_learning/data/
# rsync  -av -u --exclude *.pt  -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa" /data/JNJER/Deep_learning/data/Imagenet_bbox lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/Deep_learning/data/
# rsync  -av -u -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa" lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/Deep_learning/data/Imagenet_bbox  /Volumes/data/2024_archives/2024_science/Deep_learning/data/
# DATADIR=/data/JNJER/Deep_learning
# data_push_meso:
# 	rsync $(MESO_OPTS) $(DATADIR)/Deep_learning $(MESO_URL)

# data_pull_meso:
# 	rsync  $(MESO_OPTS) $(MESO_URL)/Deep_learning $(DATADIR)



#################@#################@#################@#################

optuna:
	optuna-dashboard sqlite:///cached_data/2024-05-14_optuna.sqlite3

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
