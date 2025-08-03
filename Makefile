DIR=Retinotopy
default:notebooks_learn
#################@#################@#################@#################
# 
MESO_URL=lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/science/
MESO_URL=lperrinet@193.51.217.241:science

SSH_OPTS_MESO=-av -u --info=progress2 --exclude pytorch.sif --exclude ._n* -e "ssh -p 8822 -i ~/.ssh/id-ring-ecdsa"
pull_meso:
	rsync $(SSH_OPTS_MESO) $(MESO_URL)/$(DIR)/cached_data .

push_meso:
	rsync $(SSH_OPTS_MESO) cached_data $(MESO_URL)/$(DIR) 

# cd /envau/work/neopto/USERS/PERRINET/Retinotopy 
ENVAU_URL=perrinet.l@niolon.intlocal.univ-amu.fr:/envau/work/neopto/USERS/PERRINET
SSH_OPTS=-av -u --info=progress2 --exclude .DS_Store --exclude pytorch.sif --exclude ._*  --exclude log_* -e "ssh -i ~/.ssh/id-ring-ecdsa"
pull_envau:
	rsync $(SSH_OPTS) $(ENVAU_URL)/$(DIR)/cached_data .

push_envau:
	rsync $(SSH_OPTS) cached_data $(ENVAU_URL)/$(DIR) 

JEANZAY_URL=uvb28bo@jean-zay3.idris.fr:/lustre/fswork/projects/rech/fsx/uvb28bo# $WORK http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-calculateurs-disques-eng.html
pull_jeanzay:
	rsync $(SSH_OPTS) $(JEANZAY_URL)/$(DIR)/cached_data .

push_jeanzay:
	rsync $(SSH_OPTS) cached_data $(JEANZAY_URL)/$(DIR) 

data_pull_envau_local:
	rsync $(SSH_OPTS) $(ENVAU_URL)/Deep_learning/data/Imagenet_{full,bbox} data
	rsync $(SSH_OPTS) $(ENVAU_URL)/Deep_learning/data/animal_10k  data

data_pull_jeanzay:
	rsync $(SSH_OPTS) $(JEANZAY_URL)/DeepLearningDatasets/Imagenet_{full,bbox} /envau/work/neopto/USERS/PERRINET/Deep_learning/data
	rsync $(SSH_OPTS) $(JEANZAY_URL)/DeepLearningDatasets/animal_10k  /envau/work/neopto/USERS/PERRINET/Deep_learning/data

# to make from niolon
data_push_jeanzay:
	rsync $(SSH_OPTS) /envau/work/neopto/USERS/PERRINET/Deep_learning/data/Imagenet_{full,bbox}  $(JEANZAY_URL)/DeepLearningDatasets
	rsync $(SSH_OPTS) /envau/work/neopto/USERS/PERRINET/Deep_learning/data/animal_10k  $(JEANZAY_URL)/DeepLearningDatasets

# rsync -av perrinet.l@niolon.intlocal.univ-amu.fr:/envau/work/neopto/USERS/PERRINET/Retinotopy/cached_data .

DATADIR=data
DATAPATH=/Volumes/SSD1TO/Deep_learning/data
data_push_meso:
	rsync $(SSH_OPTS) --dry-run $(DATAPATH)/Imagenet_* $(MESO_URL)/$(DIR)/$(DATADIR)

data_pull_meso:
	rsync $(SSH_OPTS) --delete $(MESO_URL)/$(DIR)/$(DATADIR)/Imagenet_* $(DATAPATH)
#################@#################@#################@#################
J=jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
# JN=$(J) --to markdown --stdout # for dev
JN=$(J) --to notebook --inplace # for the final touch

notebooks_intro:
	# ipython 00_installation.ipynb                 
	# $(JN)  01_retinotopic-mapping.ipynb         
	$(JN) 04_display_dataset_ground_truth.ipynb 
	$(JN) 05_imagenet_boxes_dataset.ipynb       
# 	$(JN) 08_dataloaders.ipynb                  
	$(JN) 09_benchmark-dataloader.ipynb
	# find -s . -name "0*.ipynb" -exec $(JN) {} \;

notebooks_learn:
	ipython 10_transfer_learning.ipynb 
	ipython 14_optimise.ipynb	
	ipython 16_rotation_attack.ipynb
	ipython 17_zoom_attack.ipynb
	ipython 18_translation_attack.ipynb
# 	find -s . -name "1*.ipynb" -exec $(JN) {} \;
	
notebooks_maps:
	ipython 21_multiple_likelihood_map.ipynb 22_scan_mean_nan.ipynb 25_stats_of_likelihood_map.ipynb 26_stats_likelihood_map_Animal10K.ipynb 28_grad_cam_complete_evaluation.ipynb
# 	find -s . -name "2*.ipynb" -exec $(JN) {} \;

all:
	# find -s . -name "*.ipynb" -exec ls -ltr {} \;
	find -s . -name "*.ipynb" -exec $(JN) {} \;

#################@#################@#################@#################

optuna:
	optuna-dashboard sqlite:///cached_data/2025-06-20_optuna.sqlite3

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
#################@#################@#################@#################

clean:
	find /Volumes/SSD1TO/Deep_learning/data/Imagenet_bbox -type f -name ._n\* -delete
	find /Volumes/SSD1TO/Deep_learning/data/Imagenet_full -type f -name ._n\* -delete

#################@#################@#################@#################
