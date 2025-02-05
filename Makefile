DIR=Retinotopy
#################@#################@#################@#################
# 
MESO_URL=lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/
MESO_URL=lperrinet@193.51.217.241:science

MESO_OPTS=-av -u --info=progress2 --exclude pytorch.sif --exclude ._n*  -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa"
pull_meso:
	rsync $(MESO_OPTS) $(MESO_URL)/$(DIR)/cached_data .

push_meso:
	rsync  $(MESO_OPTS) cached_data $(MESO_URL)/$(DIR) 


DATADIR=data
DATAPATH=/Volumes/SSD1TO/Deep_learning/data
data_push_meso:
	rsync $(MESO_OPTS) --dry-run $(DATAPATH)/Imagenet_* $(MESO_URL)/$(DIR)/$(DATADIR)

data_pull_meso:
	rsync  $(MESO_OPTS) --delete $(MESO_URL)/$(DIR)/$(DATADIR)/Imagenet_* $(DATAPATH)

#################@#################@#################@#################

optuna:
	optuna-dashboard sqlite:///cached_data/2025-01-05_optuna.sqlite3

#################@#################@#################@#################

load_modules:
	module load userspace/all; module load cuda/10.1

# https://arcca.github.io/intro_singularity/06-singularity-gpu/index.html
singularity_build:
	singularity build pytorch.sif pytorch.def

singularity:
	singularity shell --bind /scratch:/scratch --nv pytorch.sif

#################@#################@#################@#################
venv:
	python3 -m venv venv
	source .venv/bin/activate
	
update:
	pip install --upgrade -r requirements.txt
