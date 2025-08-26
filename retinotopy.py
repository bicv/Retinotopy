#############################################################
# data_set_type = 'focus' # Select your root between : 'boxes', 'square', 'focus', 'full', 'square'
# data_set_types = ['raw', 'full', 'bbox']
data_set_types = ['full', 'bbox']
data_set_linestyles = [':', '-.', '-', ]
import os
import platform
HOST = platform.uname()[1]

# print(f'{HOST=}')
def touch(fname): open(fname, 'w').close()
# import requests
import math
import time
from time import strftime, gmtime
datetag = strftime("%Y-%m-%d", gmtime())
#datetag = '2024-05-24'
# datetag = '2025-01-05'
datetag = '2025-03-06' # Jean Zay
datetag = '2025-05-08' # Jean Zay
datetag = '2025-06-20' # Jean Zay

#############################################################

#############################################################
# MATPLOTLIB imports and parameters
import numpy as np
import json
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
subplotpars = SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

# matplotlib parameters
from matplotlib import font_manager
# Fig variables
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue",  "lightsteelblue", "lavender", "white", "seashell", "mistyrose",  "firebrick"])

fig_width = 20
fontsize = 14
font = font_manager.FontProperties(weight='normal', size=fontsize)
dpi = 'figure'
dpi = 200
opts_savefig = dict(dpi=dpi, bbox_inches='tight', pad_inches=0, edgecolor=None)

colors = ['b', 'r', 'k', 'g', 'm', 'y']
phi = (np.sqrt(5)+1)/2 # golden ratio for the figures :-)

#to plot & display 
def pprint(message:str): 
    "Display function for strings"
    print('-'*len(message))
    print(message)
    print('-'*len(message))

def transparent_cmap(cmap:matplotlib.colors.ListedColormap, N:int=255):
    "Copy colormap and set alpha values so its transparent at lower bound"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 1, N+4, endpoint=True)
    return mycmap



# HACK to rename files
# import glob
# for old_file_name in glob.glob('cached_data/2024-05-14*'):
#     new_file_name = old_file_name.replace('do_polar=True', 'retino').replace('do_polar=False', 'cartesian')
#     print(old_file_name, new_file_name)
#     os.rename(old_file_name, new_file_name)

def get_filename(data_cache:str, datetag:str, data_set_type:str, model_name:str, do_polar:bool):
    """Generate a string filename based on the datetag, data, model 
    or transformation used during the process"""
    return f"{data_cache}/{datetag}_{data_set_type}_{model_name}_{'retino' if do_polar else 'cartesian'}"

exts = ['pdf', 'png']
#############################################################

#############################################################
# Importing libraries
import pandas as pd # to store results
import torch
import torch.nn.functional as nnf
import torchvision
from torchvision.io import read_image
# https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
from torchvision.transforms.functional import crop
# from torchvision import datasets, models, transforms
# from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
import torch.nn as nn
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

if torch.backends.mps.is_available():
    device = torch.device('mps')
    # device = torch.device('cpu') # HACK to avoid using MPS on macOS silicon chips
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU : ', torch.cuda.get_device_name(), '#GPU=', torch.cuda.device_count())    
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# set seed function
def set_seed(seed=None, seed_torch:bool=True, verbose:bool=False):
  "Define a random seed or use a predefined seed for repeatability"
  if seed is None:
    seed = np.random.choice(2 ** 32)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  if verbose: print(f'Random seed {seed} has been set.')    

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f">> Scanning variables occupying GPU memory...")
    for var_name, var in globals().items():
        if torch.is_tensor(var) and var.is_cuda:
            print(f"{var_name}: {var.element_size() * var.nelement() / 1024**2:.2f} MB")

    
#############################################################

#############################################################
data_cache = 'cached_data'
os.makedirs(data_cache, exist_ok=True)

interpolation = T.InterpolationMode.BILINEAR
padding_mode = "border"

batch_size = 50 # Set the batch size for training and validation

USER = os.environ['USER']  # username

# platform-dependent variables
if USER=='uvb28bo': # Jean Zay
    # work_dir = os.environ['WORK']
    # DATAROOT = f'{work_dir}/DeepLearningDatasets' 
    # DATAROOT = f'{os.environ['SCRATCH']}/data'
    DATAROOT = f'/lustre/fsn1/projects/rech/fsx/uvb28bo/data' 
    # num_workers = 16 # on H100
    num_workers = 8 # on V100
    print(f'Running on Jean Zay with {torch.cuda.get_device_name()} with {DATAROOT=} and {USER=} ')
# elif '.cluster' in HOST: # mesocentre
#     DATAROOT = '/scratch/lperrinet/science/Deep_learning/data'
#     num_workers = 8
elif 'm-gpu' in HOST: # MESONET
    DATAROOT = 'data'
    num_workers = 16
# elif HOST in ['babbage']: # 
#     DATAROOT = '/data/Deep_learning/data'
#     num_workers = 2
# elif HOST in ['CONECT-LID-01']: # emmy
#     # DATAROOT = '/envau/userspace/perrinet.l/data'
#     DATAROOT = '/scratch/data'
#     batch_size = 256
#     num_workers = 8    
# elif HOST in ['CONEC-LID-002']: # faraday
#     # DATAROOT = '/envau/userspace/perrinet.l/data'
#     DATAROOT = '/scratch/ImageNet'
#     num_workers = 16    
# elif HOST in ['inv-ope-de06', 'INV-133-DE01']: # CURIE , ada
#     DATAROOT = '/data/JNJER/Deep_learning/data'
#     num_workers = 2
elif HOST in ['neo-ope-de04']: # Darwin  
    DATAROOT = '/data/JNJER/Deep_learning/data'
    num_workers = 16
# elif HOST in ['brain-lid-004']: # GPU manu  
#     DATAROOT = '/data/JNJER/Deep_learning/data'
#     num_workers = 16
# elif 'obiwan' in HOST: 
#     # DATAROOT = '/Volumes/UnaTera/2023_archives/2023_science/JNJER_PhD/data'
#     DATAROOT = '/Volumes/data/2024_archives/2024_science/Deep_learning/data'
#     DATAROOT = '/Volumes/SSD1TO/ImageNet'
#     DATAROOT = '/Volumes/SSD1TO/Deep_learning/ILSVRC2010_ImageNet'
#     DATAROOT = 'data'
#     DATAROOT = '/Volumes/SSD1TO/DeepLearningDatasets'
#     interpolation = T.InterpolationMode.NEAREST
#     padding_mode = "reflection"
#     num_workers = 2
elif 'Ahsoka' in HOST: 
    DATAROOT = '/Volumes/backups/2023_archives/2023_science/JNJER_PhD/data'
    DATAROOT = '/Volumes/data/2024_archives/2024_science/Deep_learning/data'
    num_workers = 24
    device = torch.device('cpu')
elif 'DESKTOP-27VNO0E' in HOST: 
    DATAROOT = 'd:\\Data'
    batch_size = 50
    num_workers = 8
elif 'Newton' in HOST: 
    if os.path.isdir('/media/jnjer/Transcend/Data'):
        DATAROOT = '/media/jnjer/Transcend/Data'
    else:
        DATAROOT = 'c:\\Users\\JnJer\\Nextcloud\\JNJER_PhD\\data'
    batch_size = 50
    num_workers = 4
else:
    raise ValueError(f'Unknown host {HOST=} / {USER=}')

#############################################################
def welcome():
    pprint(f'On date {datetag}, Running learning on host {HOST} with device {device}, pytorch=={torch.__version__}')
    print('Welcome on', platform.platform())
#############################################################
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass, asdict, field

@dataclass
class Params:
    
    datetag: str = datetag # Set the date of the result's file
    loader: str = 'data/Imagenet_urls_ILSVRC_2016.json' # File containing Imagenet's labels
    annotations_animal: str = 'data/Animal10k_annotations.json' # File containing Animak10k's labels
    annotations_train: str = 'data/LOC_train_solution.csv' # File containing Imagenets's labels
    annotations_val: str = 'data/LOC_val_solution.csv' # File containing Imagenets's labels

    # root: str = f'{DATAROOT}/Imagenet_{data_set_type}' # Directory containing images to perform the training
    folders: list = field(default_factory=lambda: ['val', 'train']) # Set the training and validation folders relative to the root
    tasks: list = field(default_factory=lambda: ['animal', 'dog', 'cat', 'bird']) # Set the semantic link to perfome different tasks
    
    image_size: int = 224 # base resolution of the image (224, 224)
    num_epochs: int = 20 # 
    n_train_stop: int = 0 # set to zero to use all images
    seed: int = 1998 # Set the seed for reproducibility 
    batch_size: int = batch_size # Set number of images per input batch
    batch_size_val: int = batch_size # Set number of images per input batch
    lr_conv: float = 1.e-5 # Set learning rate for the classification layers
    lr_class: float = 1.e-3 # Set learning rate for the classifier layers
    mutnemom: float = 0.1 # Set the momentum = 1 - mutnemom
    ateb2: float = 0.001 # Sets the second momentum as beta2 = 1 - ateb2 or use SGD if it is set to 0
    weight_decay: float = 0.01 # See https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    label_smoothing: float = 0.01 # See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    rs_min: float = 0.00 # Set minimum radius of the log-polar grid
    rs_max: float = -5.00 # Set maximum radius of the log-polar grid
    
    do_polar: bool = True # use a retinotopic mapping
    do_raw: bool = False # use pytorch pre-trained weights and image transform
    do_translate: bool = False # use translation attacks
    do_resize: bool = True # resize the image to args.image_size
    do_mask: bool = True # add a circular mask on the Cartesian input to match the retino input (circular window) 
    do_scratch: bool = False # whether we use pretrained weights or not during transfer learning
    do_rotation: bool = False # just use this for rotation attacks
    resolution: tuple = (11, 11) # resolution of the likelihood map
    size_ratio: float = 0.1 # how much of the image to use relative to radius
    do_saccade: bool = False # True to get multiple pov for eah image in the data set transform
    do_zoom: bool = False # True to apply a zoom (range from args.size_ratio to (2 + args.size_ratio) +/- 0.1 )
    method: str = 'valid' #select sampling for mapping between full = with border & valid = no border
    saccade_type: str = 'multi' #select sampling for mapping between multi = with multiple ratio & grid = same sample ratio 
    angles = np.linspace(-180, 180, 100)   # combination of angles used for training or attacks
    normalize: bool = True # use pytrorch normalization for imagenet
    
    verbose: bool = False
    if verbose: welcome()
    set_seed(seed=seed, seed_torch=True, verbose=verbose)
    
args = Params()

# keep a human readable copy of the parameters
json_fname = os.path.join(data_cache, datetag + '_config_args.json')
if not(os.path.isfile(json_fname)):
    with open(json_fname, 'wt') as f:
        json.dump(vars(args), f, indent=4)
#############################################################


#############################################################
# Wordnet
from nltk.corpus import wordnet as wn

#----------------Get the label for the Imagenet categorization------------------------

def get_labels(args:dict, all_refs:bool=False):   
    """ Get the labels for the Imagenet data set
    based on the json file containing ground true labels
    match : dictionary with the task as key and the list corresponding label as value
    labels : list of labels for the data set
    revlabels_dico : dictionary with the synset as key and the label as value
    labels_dico : dictionary with the label as key and the synset as value
    i_labels_dico : dictionary with the label as key and the index of the label in the list as value
    if all_refs is True, return all the references for the labels
    """

    with open(args.loader) as json_file:
        Imagenet_urls_ILSVRC_2016 = json.load(json_file)

    match = {}
    labels = []

    if all_refs:
        revlabels_dico = {}
        labels_dico = {}
        i_labels_dico = {}

    for task in args.tasks:
        match[task] = []

    for i_id, img_id in enumerate(Imagenet_urls_ILSVRC_2016):
        syn_= wn.synset_from_pos_and_offset('n', int(img_id.replace('n',''))) # synset for the label
        sem_ = syn_.hypernym_paths()[0]
        label = syn_.name().split(".")[0]
        labels.append(label)
        if all_refs:
            i_labels_dico[label] = i_id
            labels_dico[label] = img_id
            revlabels_dico[img_id] = label
        for i in np.arange(len(sem_)):
            for task in args.tasks:
                if sem_[i].lemmas()[0].name() in task :
                    match[task].append(i_id)
    
    if all_refs:
        return match, labels, revlabels_dico, labels_dico, i_labels_dico  
    else:
        return match, labels

# match, labels = get_labels(args)
match, labels, revlabels_dico, labels_dico, i_labels_dico = get_labels(args, all_refs=True)
#---------------Get annotations from other the data set (Animal10k, ...)---------------
def get_annotation(type:str):
    """ Return the correct data set annotation
    based on the format:
    - csv for Imagenet
    - json for Aniaml10k
    """
    if 'csv' in type:
        with open(args.annotations_val, 'r') as csv_file:
            return pd.read_csv(csv_file)
    else:
        return json.load(open(args.annotations_animal)) 
    
#############################################################


#############################################################
# https://github.com/laurentperrinet/2024-12-09-normalizing-images-in-convolutional-neural-networks
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225]) 

def imgs_to_np(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std):
    images = torchvision.utils.make_grid(img_list, nrow=11)
    """Imshow for Tensor."""
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)

def imshow(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std, 
           title:str=None, fig_height:int=7, fig=None, ax:matplotlib.axes.Axes=None, save:bool=False, name:str=None): #allow to display the input image
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_height*len(img_list), fig_height))
    inp = imgs_to_np(img_list, im_mean=im_mean, im_std=im_std)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title != None: fig.suptitle(title)
    fig.set_facecolor(color='white')
    #plt.tight_layout()

    if save:
        to_save(fig, name=name)
    
def to_dev(args:dict, item:any):
    """Move the item to the device specified in args."""
    if hasattr(args, 'device'):
        return item.to(args.device)
    else:
        return item
    
def get_start(start_value:float=0.1):
    """Get the start value for log2 based on the ratio."""
    return torch.log2(torch.tensor(start_value)).item()

def get_grid(args:dict, endpoint:bool=False):
    """Generate a grid for the log-polar mapping
    """

    rs_ = torch.logspace(args.rs_min, args.rs_max, args.image_size, base=2) # Radial distances (log scale)
    if endpoint: ## If endpoint is True, include the last point in the range
        ts_ = torch.linspace(0, torch.pi*2, args.image_size) # Angular positions (full circle)
    else:
        ts_ = torch.linspace(0, torch.pi*2, args.image_size+1)[:-1] 
    grid_xs = torch.outer(rs_, torch.cos(ts_)) # X-coordinates
    grid_ys = torch.outer(rs_, torch.sin(ts_)) # Y-coordinates	
    
    return to_dev(args, torch.stack((grid_xs, grid_ys), 2)) # (H_scaled, W_scaled, 2)

def get_zoom_grid(args:dict):
    """
    Generate grids for `nnf.grid_sample` where each ratio is a different zoom level.

    Args:
        rs_max (float): Maximum radius for the log-polar grid.
        ratio (float): Ratio for the zoom level.
        image_size (int): Size of the image (height and width).

    Returns:
        torch.Tensor: A tensor of shape [n_fixation_points, C, W, H], 
        where each fixation point is a zoomed version of the image.
        
    """
    grids = []
    for ratio in np.arange(args.size_ratio, (10 + args.size_ratio), 0.1): 
        if args.do_polar: ## Generate log-polar grid
            start = get_start(ratio) # Get the start value for log2 based on the ratio
            rs_ = torch.logspace(start, args.rs_max, args.image_size, base = 2)
            ts_ = torch.linspace(0, torch.pi*2, args.image_size+1)[:-1]  
        
            grid_x = torch.outer(rs_, torch.cos(ts_)) 
            grid_y = torch.outer(rs_, torch.sin(ts_)) 
            
        else: ## Generate Cartesian grid
            x = torch.linspace(-ratio, ratio, args.image_size) # X-coordinates based on the ratio
            y = torch.linspace(-ratio, ratio, args.image_size) # Y-coordinates based on the ratio
            grid_y, grid_x = torch.meshgrid(x, y, indexing='ij')

        
        grids.append(torch.stack((grid_x, grid_y), 2))
        
    return torch.stack(grids)

def generate_translate_roll_grids(args:dict):
    """
    Generate grids for `nnf.grid_sample` where each fixation point rolls the image
    without losing information, mimicking `np.roll` behavior.
    
    Args:
        image_shape (tuple): Shape of the original image as (C, H, W).
        n_fixation_points (int): Number of fixation points along each axis.

    Returns:
        torch.Tensor: A tensor of shape [n_fixation_points^2, C, W, H],
                      with each fixation point shifted and wrapped around.
    """
    
    if args.do_polar:
        # Generate base log-polar grid
        start = get_start(1)
        rs_ = torch.logspace(start, args.rs_max, args.image_size, base=2)  # Radial distances (log scale)
        ts_ = torch.linspace(0, torch.pi * 2, args.image_size + 1)[:-1]  # Angular positions (full circle)

        grid_x = torch.outer(rs_, torch.cos(ts_))
        grid_y = torch.outer(rs_, torch.sin(ts_))
        base_grid = torch.stack((grid_x, grid_y), dim=2)  # (H_scaled, W_scaled, 2)
        # Normalize the log-polar grid to range [-1, 1]
        base_grid[..., 0] = 2 * (base_grid[..., 0] - base_grid[..., 0].min()) / (base_grid[..., 0].max() - base_grid[..., 0].min()) - 1
        base_grid[..., 1] = 2 * (base_grid[..., 1] - base_grid[..., 1].min()) / (base_grid[..., 1].max() - base_grid[..., 1].min()) - 1
    else:
        grid_y = torch.linspace(-1, 1, args.image_size)
        grid_x = torch.linspace(-1, 1, args.image_size)
        base_grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing='ij'), dim=-1)  # (H, W, 2)

        
        base_grid[..., 0] = -base_grid[..., 0]
        base_grid = base_grid[..., [1, 0]]  # Swap x and y axes
        base_grid[..., 1] = -base_grid[..., 1]  # Invert the new y-axis (originally x-axis)

    # Create linear fixation points
    fixation_points_x = torch.linspace(1, -1, args.resolution[0])
    fixation_points_y = torch.linspace(1, -1, args.resolution[1])
    fixation_grid = torch.stack(torch.meshgrid(fixation_points_x, fixation_points_y, indexing='ij'), dim=-1)

    roll_grid = []
    for point in fixation_grid.view(-1, 2):
        shifted_grid = base_grid.clone()
        shifted_grid[..., 0] -= point[1]  # Shift x-axis (horizontal)
        shifted_grid[..., 1] -= point[0]  # Shift y-axis (vertical)

        # Wrap around grid values to simulate rolling
        shifted_grid[..., 0] = torch.remainder(shifted_grid[..., 0] + 1, 2) - 1
        shifted_grid[..., 1] = torch.remainder(shifted_grid[..., 1] + 1, 2) - 1

        roll_grid.append(shifted_grid)

    return torch.stack(roll_grid, dim=0)


def get_saccade_map(args:dict):
    """
    Generate grids for `nnf.grid_sample` where each fixation point is a different saccade.
    it differs from the translation grid in that it crops the image at the same time.
    
    Args:
        size_ratio (float): Ratio for the zoom level.
        rs_max (float): Maximum radius for the log-polar grid.
        image_size (int): Size of the image (height and width).
        method (str): Method for sampling ('valid' stays in the borders or 'full' spills over the edges).
        resolution (tuple): Resolution of the grid (number of fixation points along each axis).
        do_polar (bool): Whether to use polar coordinates or Cartesian coordinates.
    returns:
        torch.Tensor: A tensor of shape [n_fixation_points, C, W, H],"""
    grids = []
    if args.do_polar:
        start = get_start(args.size_ratio)
        rs_ = torch.logspace(start, args.rs_max, args.image_size, base = 2)
        ts_ = torch.linspace(0, torch.pi*2, args.image_size+1)[:-1]  
    
        grid_x = torch.outer(rs_, torch.cos(ts_)) 
        grid_y = torch.outer(rs_, torch.sin(ts_)) 
        
    else:
        x = torch.linspace(-args.size_ratio, args.size_ratio, args.image_size)
        y = torch.linspace(-args.size_ratio, args.size_ratio, args.image_size)
        grid_y, grid_x = torch.meshgrid(x, y, indexing='ij')
        
    border = .9 if args.method == 'full' else 1-args.size_ratio

    if max(args.resolution) == 1:
        return torch.stack((grid_x, grid_y), 2).unsqueeze(dim=0)
    
    for i in np.linspace(-border, border, (args.resolution[0])):
        for j in np.linspace(-border, border, (args.resolution[1])):
            grids.append(torch.stack((grid_x+j, grid_y+i), 2))
    
    return torch.stack(grids)

def round_up(num, denum):
    """Round up the number to the nearest upper integer that is a
    multiple of denum."""
    return num//denum + num%denum


def multi_sacade_map(args:dict):
    """Generate grids for `nnf.grid_sample` where each fixation point is a different saccade.
    it differs from the saccade map grid in that it uses a different ratio for each fixation point
    Args:
        all_ratios (list): List of ratios for the multiple saccade map.
        rs_max (float): Maximum radius for the log-polar grid.
        image_size (int): Size of the image (height and width).
        method (str): Method for sampling ('valid' stays in the borders or 'full' spills over the edges).
        resolution (tuple): Resolution of the grid (number of fixation points along each axis).
        do_polar (bool): Whether to use polar coordinates or Cartesian coordinates.
    returns:
        torch.Tensor: A tensor of shape [n_fixation_points, C, W, H],
    """
    all_ratios = np.linspace(1, args.size_ratio, round_up(args.resolution[0],2))
    grids_saccades = []
    for num, i in enumerate(np.linspace(1, args.resolution[0], round_up(args.resolution[0],2), dtype=int)):
        #if args.saccade_type == 'multi':
        #    args.method = 'full' if len(grids_saccades) == len(all_ratios)-1 else 'valid'
        args.resolution = (i,i)
        args.size_ratio = all_ratios[num]
        grids_saccades.append(get_saccade_map(args).reshape(args.resolution[0], args.resolution[1],
                                                                args.image_size, args.image_size, 2))

    for i in np.linspace(0, len(grids_saccades)-2, len(grids_saccades)-1, dtype=int):
        grids_saccades[i+1][1:-1,1:-1] = grids_saccades[i]
    
    return grids_saccades[i+1].reshape(args.resolution[0] * args.resolution[1],
                                                                args.image_size, args.image_size, 2)

def apply_grid(image:torch.tensor, grid:torch.tensor):
    """Apply a grid to an image using grid_sample."""
    image = image.repeat(grid.shape[0],1,1,1)
    return nnf.grid_sample(image, grid, 
                            padding_mode=padding_mode, align_corners=False).squeeze(dim=0)

class transform_apply_grid(object): 
    def __init__(self, logPolar_grid, mode):
        self.grid = logPolar_grid
        self.mode = mode

    def __call__(self, images):
        if self.mode == 'base':
            return nnf.grid_sample(images.unsqueeze(dim=0), self.grid.unsqueeze(dim=0), 
                               padding_mode="zeros", align_corners=False).squeeze(dim=0)
        if self.mode == 'multiple':
            return nnf.grid_sample(images.unsqueeze(dim=0).repeat(len(self.grid),1,1,1), self.grid,                                    
                               padding_mode="zeros", align_corners=False).squeeze(dim=0)
        else:
            return nnf.grid_sample(images, self.grid, 
                                padding_mode="zeros", align_corners=False).squeeze(dim=0)

    
def make_mask(image_size:int, radius:float = 0.5):
    """Create a circular mask for the image.
    image_size: int, size of the image (height and width)
    radius: float, radius of the circle (0.5 means half the image size)"""
    X, Y = np.meshgrid(np.linspace(-radius, radius, image_size, endpoint=True), 
               np.linspace(-radius, radius, image_size, endpoint=True))
    R = np.sqrt(X**2 + Y**2)
    mask = (R < 0.5).astype(np.float32)
    return torch.from_numpy(mask)

class ApplyMask: 
    """Apply a mask to the image."""
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, images):
        return images[:, :, ::] * self.mask

class CleanRotations_class(object): 
    """Apply a rotation to the image.
    Clean because apply a circular mask to the image so the border does not reveal the rotation."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, images):
        temp = []
        images = images.unsqueeze(dim=0) if len(images) == 3 else images
        for image in images:
            for angle in self.angles:
                temp.append(T.functional.rotate(image, angle=angle, expand = False)) 
        return torch.stack(temp)

def CleanRotations_function(image, mask, angles=[0]):
    temp = []
    for angle in angles:
        temp.append(T.functional.rotate(image, angle=angle, expand = False))
    temp = torch.stack(temp)
    return temp[:,:,::]*torch.from_numpy(mask)#.to(device)
    

# Resnet datasets initialisation
def get_transforms(args:dict, im_mean:np.array=im_mean, im_std:np.array=im_std, do_augment:bool=True):
    """Get the transforms for the image data set."""
    
    transforms = [                
        T.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        T.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    ]

    if do_augment: # apply data augmentation to the image
        transforms.append(T.RandomHorizontalFlip())
        # transforms.append(T.RandomCrop())
        # transforms.append(T.AutoAugment())
        transforms.append(T.TrivialAugmentWide())  # Doesn't include rotation by default

    if args.do_rotation and not args.do_saccade: # apply rotation to the image
        args.batch_size_val, args.batch_size = 1, 1
        grid = get_grid(args).repeat(len(args.angles), 1, 1, 1)
        transforms.append(CleanRotations_class(args.angles))

    if args.do_zoom and not (args.do_saccade or args.do_translate): # apply zoom to the image
        args.batch_size_val, args.batch_size = 1, 1
        grid_zoom = to_dev(args, get_zoom_grid(args))
        transforms.append(transform_apply_grid(grid_zoom, 'multiple'))
        
        if not args.do_polar: # if we are not using polar coordinates, we need to apply a mask to the image
            mask = to_dev(args, make_mask(args.image_size))
            transforms.append(ApplyMask(mask))

    if args.do_translate and not (args.do_saccade or args.do_zoom): # apply translation to the image
        args.batch_size_val, args.batch_size = 1, 1
        grid_translate = to_dev(args, generate_translate_roll_grids(args))
        transforms.append(transform_apply_grid(grid_translate, 'multiple'))


    if args.do_polar and not (args.do_saccade or args.do_zoom or args.do_translate): # apply log-polar mapping to the image
        grid_polar = get_grid(args) if not args.do_rotation else get_grid(args).repeat(len(args.angles), 1, 1, 1)
        transforms.append(transform_apply_grid(grid_polar, ('base' if not args.do_rotation else None)))

    if args.do_resize and not (args.do_polar or args.do_saccade or args.do_zoom): # resize the image to args.image_size
        transforms.append(T.Resize(int(args.image_size), interpolation=interpolation, antialias=True))
        transforms.append(T.CenterCrop((int(args.image_size), int(args.image_size))))
        
    if args.do_mask and not (args.do_polar or args.do_saccade or args.do_zoom): # apply a circular mask to the image
        mask = to_dev(args, make_mask(args.image_size))
        transforms.append(ApplyMask(mask))

    if args.do_saccade : # apply saccade to the image
        args.batch_size_val = 1
        grid = to_dev(args, multi_sacade_map(args)) if args.saccade_type == 'multi' else to_dev(args, get_saccade_map(args)) 
        transforms.append(transform_apply_grid(grid, 'multiple'))

        if not args.do_polar and not args.do_raw: # if we are not using polar coordinates, we need to apply a mask to the image
            mask = to_dev(args, make_mask(args.image_size))
            transforms.append(ApplyMask(mask))

    if args.normalize:
        transforms.append(T.Normalize(mean=im_mean, std=im_std)) # to normalize colors on the imagenet dataset
    
    return T.Compose(transforms)

from torchvision.datasets import ImageFolder

def is_valid_file(path:str):
    """
    Filter out files starting with '._'
    
    Args:
        path (str): Full path to the file
    
    Returns:
        bool: True if the file should be included, False otherwise
    """
    # Get the filename from the full path
    filename = os.path.basename(path)
    
    # Return False if filename starts with '._'
    if filename.startswith('._'):
        return False
    
    return True


def image_datasets_transforms(args:dict, im_mean:np.array=im_mean, im_std:np.array=im_std, verbose:bool=True):
    """
    Load the image data set and apply the transforms to it.
    return a dictionary with the folder name as key and the data set as value.
    The data set is a pytorch ImageFolder object.
    """

    image_datasets  = {}
    for folder in args.folders:

        data_transform = get_transforms(args, im_mean=im_mean, im_std=im_std, 
                                        do_augment=(folder=='train' and not args.do_raw))

        # load the data
        path = os.path.join(args.root, folder) # data path
        image_datasets[folder] = ImageFolder(path, 
                                             transform=data_transform,
                                             is_valid_file=is_valid_file)

        if verbose: 
            print(f"Loaded {len(image_datasets[folder])} images under {folder}")  

    return image_datasets


def datasets_transforms(args:dict, im_mean:np.array=im_mean, im_std:np.array=im_std,
                        num_workers:int=num_workers, pin_memory:bool=True, shuffle:bool=True, verbose:bool=True):
    """
    Load the image data set and apply the transforms to it.
    return a dictionary with the folder name as key and the data set as value.
    The data set is a pytorch DataLoader object.
    """

    image_datasets  = image_datasets_transforms(args, im_mean=im_mean, im_std=im_std, verbose=verbose)

    dataloaders = {}
    for folder in args.folders:

        dataloaders[folder] = torch.utils.data.DataLoader(
                                image_datasets[folder], 
                                batch_size=args.batch_size if folder=='train' else args.batch_size_val,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
                        )
    return dataloaders


#############################################################

#############################################################
def make_padding_circular_again(model_retrain):
    """
    Make the padding circular for the model.
    
    This is needed for the retraining of the model in log-polar coordinates.
    TODO: test if that helps improving the accuracy of the model.

    """
    for child in list(model_retrain.children()):
        if isinstance(child, (nn.Conv2d)):
            child.padding_mode = 'circular'
        for grandchild in list(child.children()):
            if isinstance(grandchild, (nn.Conv2d)):
                grandchild.padding_mode = 'circular'
            for grandgrandchild in list(grandchild.children()):
                if isinstance(grandgrandchild, (nn.Conv2d)):
                    grandgrandchild.padding_mode = 'circular'
                for grandgrandgrandchild in list(grandgrandchild.children()):
                    if isinstance(grandgrandgrandchild, (nn.Conv2d)):
                        grandgrandgrandchild.padding_mode = 'circular'

    return model_retrain

def train_model(args:dict, model, dataloaders:torch.utils.data.DataLoader, df_train=None, each_steps:int=64, 
                verbose:bool=True, do_save:bool=True, model_filename='resnet.pt'):
    
    # retraining the full model
    for param in model.parameters():
        param.requires_grad = True        

    backbone_params = []
    for name, param in model.named_parameters():
        if 'fc' not in name: backbone_params.append(param)

    # Parameters for the classifier (final fully connected layer)
    classifier_params = list(model.fc.parameters())

    params = [
        {'params': backbone_params, 'lr': args.lr_conv},  # Pretrained layers
        {'params': classifier_params, 'lr': args.lr_class}  # New classifier
    ]

    # sets the optimizer
    if args.ateb2 > 0.: 
        optimizer = torch.optim.AdamW(params, betas=(1-args.mutnemom, 1-args.ateb2), weight_decay=args.weight_decay) 
    else:
        optimizer = torch.optim.SGD(params, momentum=1-args.mutnemom, weight_decay=args.weight_decay) # to set training variables
    
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing) # binary_cross_entropy_with_logits

    # the DataFrame to record from
    if df_train is None:
        i_epoch_start = 0
        df_train = pd.DataFrame([], columns=['epoch', 'i_image', 'total_image', 'avg_loss', 'avg_acc', 'avg_loss_val', 'avg_acc_val', 'time']) 
    else:
        i_epoch_start = df_train['epoch'].max() + 1
        if verbose: print(f"Starting from epoch {i_epoch_start} with {len(df_train)} records")
        # # reset the index
        # df_train.reset_index(drop=True, inplace=True)
        # make a copy of the DataFrame to avoid modifying the original one
        df_train = df_train.copy()

    since = time.time()
    total_image = 0
    n_train = len(dataloaders['train'].dataset)
    n_train_stop = args.n_train_stop
    if n_train_stop==0: n_train_stop = n_train

    avg_loss_ = avg_acc_ = []
    for i_epoch in range(i_epoch_start, args.num_epochs):
        i_image = 0
        for i_step, (images, labels) in enumerate(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)
            total_image += len(images)
            i_image += len(images)
            if i_image > n_train_stop: break # early stopping

            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            optimizer.zero_grad(set_to_none=True)
            # for param in model.parameters():
            #     param.grad = None

            outputs = model(images)
             
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, dim=1)

            avg_loss_.append(loss.item() * images.size(0))
            avg_acc_.append(torch.mean((preds == labels.data)*1.).cpu().item()) # append average accuracy in the last batch

            if (i_step % (max(n_train_stop//args.batch_size//each_steps, 1))==0) or (i_step == n_train_stop-1):
                with torch.no_grad():

                    avg_loss = np.mean(avg_loss_)
                    avg_acc = np.mean(avg_acc_)

                    loss_val = 0
                    acc_val = 0
                    model = model.eval()
                    n_val = len(dataloaders['val'])
                    for _, (images, labels) in enumerate(dataloaders['val']):
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)

                        loss = criterion(outputs, labels)

                        loss_val += loss.item() * images.size(0)

                        _, preds = torch.max(outputs.data, dim=1)
                        acc_val += torch.mean((preds == labels.data)*1.).cpu().item()

                    avg_loss_val = loss_val / n_val
                    avg_acc_val = acc_val / n_val

                    df_train.loc[len(df_train)] = {'epoch': i_epoch, 'i_image':i_image, 'total_image':total_image, 'avg_loss':avg_loss, 'avg_acc':avg_acc, 'avg_loss_val':avg_loss_val, 'avg_acc_val':avg_acc_val, 'time':time.time() - since}
                    if verbose:  print(f"{model_filename} - Epoch {i_epoch}, i_image {i_image} : train= loss: {avg_loss:.4f} / acc : {avg_acc:.4f} - val= loss : {avg_loss_val:.4f} / acc : {avg_acc_val:.4f} / time:{time.time() - since:.1f}")
                avg_loss_ = avg_acc_ = []

        if do_save:
            if verbose:  print(f"Saving...{model_filename}")
            torch.save(model.state_dict(), model_filename)
            df_train.to_json(model_filename.replace('pt', 'json'), orient='index', indent=2)


    return model, df_train


def apply_weights(model, model_path:str, verbose:bool=True):
    """Apply the weights to the model.
    Args:
        model: torch model, the model to apply the weights to
        model_path: str, path to the weights file
        verbose: bool, whether to print the loading message or not
    Returns:    
        model: torch model, the model with the weights applied"""
    if verbose: print(f'loading .... {model_path}')
    # model.load_state_dict(torch.load(model_path), map_location=torch.device(device), weights_only=True)
    # model.load_state_dict(torch.load(model_path), weights_only=True)
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model

import torchvision.models as models
def load_model(model_name:str='resnet50', model_path:str=None, do_scratch:bool=False, 
               do_circular:bool=False, verbose:bool=True):
    """Load the model from the torchvision library.
    Args:
        model_name: str, name of the model to load (resnet18, resnet50, resnet101)
        model_path: str, path to the weights file
        do_scratch: bool, whether to start from a scratch model or not
        do_circular: bool, whether to make the padding circular or not
        verbose: bool, whether to print the loading message or not
    Returns:
        model: torch model, the model with the weights applied

    BEWARE: on a cluster node, internet is unreachable, so we cannot load the weights from torchvision.models.
    Instead, we load the weights from a local file.    
    """
    # get the architecture of the network
            
    if model_name=='resnet18':
        # model = models.resnet18(weights=None if do_scratch else torchvision.models.ResNet18_Weights.DEFAULT)
        model = models.resnet18(weights=None)
    elif model_name=='resnet50':
        # model = models.resnet50(weights=None if do_scratch else torchvision.models.ResNet50_Weights.DEFAULT)
        model = models.resnet50(weights=None)
    elif model_name=='resnet101':
        # model = models.resnet101(weights=None if do_scratch else torchvision.models.ResNet101_Weights.DEFAULT)
        model = models.resnet101(weights=None)
    else:
        raise ValueError(f'Unknown model {model_name}')

    if model_path is None: # we start from a scratch model
        if not(do_scratch): # we start from a pretrained model
            model = apply_weights(model, 
                          os.path.join(data_cache, f'{model_name}.pth'), verbose=verbose)
    else: # we wish to use a saved model
        model = apply_weights(model, model_path, verbose=verbose)

    if do_circular:
        model = make_padding_circular_again(model)

    return model
#############################################################


def clean_resize(args:dict, image:torch.tensor):
    """Resize the image to the image size and center crop it."""
    image = T.Resize(args.image_size, interpolation=interpolation, antialias=True)(image)
    return T.CenterCrop((int(args.image_size), int(args.image_size)))(image)
           

def get_batch(args:dict, model, full_batch:torch.tensor, size:int=100):
    """Get a light batch of images from the bigger batch 
    and apply the model to it.
    Args:      
        args: dict, arguments for the model
        model: torch model, the model to apply to the image
        full_batch: torch tensor, the full batch of images
        size: int, size of the batch to apply the model to
    Returns:        
        proba_label: torch tensor, the output of the model allong the full batch"""
    N_fixations = args.resolution[0] * args.resolution[1]
    proba_label = torch.zeros((N_fixations, 1000))
    for idx_start in np.arange(0, N_fixations, size):
        idx_stop = np.min((idx_start+size, N_fixations))
        with torch.no_grad():
            outputs = torch.nn.functional.softmax(model(full_batch[idx_start:idx_stop]), dim=1)
        proba_label[idx_start:idx_stop, :] = outputs#.detach().cpu().numpy()
    return proba_label

def get_positions(args, image):
    _, H, W = image.shape

    min_size = np.min((H, W))
    box_size = int(min_size*args.size_ratio)
    #if args.method=='valid':
    if False:
        if H < W:
            shift = (0, (W-H)/2)
        else:
            shift = ((H-W)/2, 0)

        pos_h = np.linspace(shift[0]+box_size/2, min_size+shift[0]-box_size/2, args.resolution[0], endpoint=True)
        pos_w = np.linspace(shift[1]+box_size/2, min_size+shift[1]-box_size/2, args.resolution[1], endpoint=True)
    else:
        pos_h = np.linspace(0, H, args.resolution[0]+2, endpoint=True)[1:-1]
        pos_w = np.linspace(0, W, args.resolution[1]+2, endpoint=True)[1:-1]

    pos_H, pos_W = np.meshgrid(pos_h, pos_w)

    return pos_H, pos_W, box_size
#############################################################

def to_tuple(str_size):
    """Convert a string size to a tuple of integers."""
    return (int(str_size.split(' ')[0].split('(')[1].split(',')[0]),
            int(str_size.split(' ')[1].split(')')[0]))

def get_boxes_imagenet(box_anot:list): # function to get a list of dict for each box
    """Get the boxes from the annotation file
    from a list of strings to a list of dict with the coordinates of the boxes."""
    box = []
    for i in np.arange(len(box_anot)//5) :  # iterate for each box
        box.append({'ymin' : int(box_anot[(i*5)+1]),
                     'xmin' : int(box_anot[(i*5)+2]),
                     'ymax' : int(box_anot[(i*5)+3]),
                     'xmax' : int(box_anot[(i*5)+4])})
    return box

def get_mask_from_bb(target_size:tuple, orig_size:tuple, boxes:list): # function to get the mask from the boxes
    """Get the mask from the boxes
    from a list of dict with the coordinates of the boxes
    to a mask highlighting the boxes."""
    import cv2    
    mask = np.zeros((orig_size),dtype=np.uint8) # initialize mask
    for box in boxes : 
        mask[box['ymin']:box['ymax'],box['xmin']:box['xmax']] = 1 # fill with white pixels
    return cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR).T

def store_pandas(df, df_):
    """Store the pandas dataframe row to the complete dataframe."""
    if df is None:
        return df_
    else:
        return(pd.concat([df, df_], ignore_index=True))

def normalize_array(arr:np.array):
    """Normalize a NumPy array to the range [0, 1].
    This function handles empty arrays and avoids division by zero. """
    # Ensure input is a NumPy array and check if it's non-empty
    if arr.size == 0:
        return arr  # Return the empty array as is
    
    # Convert the array to float type if it's not already
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float)
    
    # Calculate the minimum and maximum values in the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Avoid division by zero if all values in the array are the same
    if max_val == min_val:
        return np.zeros_like(arr)
    
    # Normalize the array between 0 and 1
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr


def little_box(boxe:dict, origin_size:tuple, resolution:tuple):
    """Get the coordinates of the center of the boxe in the image.
    if the boxe is too small, return the coordinates of the center of the image."""
    x_mid = (boxe['xmax'] - boxe['xmin']) + boxe['xmin']
    y_mid = (boxe['ymax'] - boxe['ymin']) + boxe['ymin']
    return (int(np.round((x_mid*resolution[0])/origin_size[0])-1), int(np.round((y_mid*resolution[1])/origin_size[1]))-1)
    

def no_axis_title(ax:matplotlib.axes.Axes, title:str):
    """Remove the axis and title from the plot."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{title}')
    return 


def get_ground_true(args:dict, image_name:str, annotations:dict, mode:str):
    """Get the ground truth for the image.
    Args:
        image_name (str): Name of the image.
        annotations (dict): Annotations for the data set.
        mode (str): Select the dataset ('Imagenet' or 'Animal10k').
    Returns:       
        ground_true_indices (np.array): Indices of the ground truth.
        ground_true (np.array): Ground truth mask.
        origin_size (tuple): Original size of the image.
    """
    # Get the ground truth for the image
    
    if mode == 'Imagenet':
        box_anot = annotations[annotations['ImageId'] == image_name]['PredictionString'].item().split(' ')
        boxes = get_boxes_imagenet(box_anot)
        origin_size = to_tuple(annotations[annotations['ImageId'] == image_name]['origin_size'].item())
        ground_true = np.round(get_mask_from_bb(args.resolution, origin_size, boxes))
    
    else: #Get ground true for Animal 10k
        import cv2

        boxes = annotations[image_name]['keypoints']
        origin_size = [annotations[image_name]['image_info']['height'], annotations[image_name]['image_info']['width']]
        brut_array = to_heatmap(boxes, origin_size)
        normalized_array = normalize_array(brut_array)
        normalized_array[normalized_array < 0.25] = 0
        ground_true = cv2.resize(normalized_array.reshape(origin_size), (args.resolution[1], args.resolution[0]), 
                                 interpolation=cv2.INTER_LINEAR) 
    
    if len(np.where(ground_true > 0)[0]) == 0 :
        try:
            for boxe in boxes:
                coord = (little_box(boxe, origin_size, args.resolution))
                ground_true[coord[1],coord[0]] = 1
        except:
                return None, np.ones(1), origin_size # return None for value we cant caculate and 1 to stop at the next condition
            
    ground_true_indices = np.where(ground_true.reshape(args.resolution[0]*args.resolution[1]) > 0)
    
    return ground_true_indices, ground_true, origin_size

def th_delete(tensor:torch.tensor, indices:list):
    """Delete the indices from the tensor."""
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def get_IoU(heatmap, ground_true):
    """Calculate the Intersection over Union (IoU) between the heatmap and ground truth.
    Args:   
        heatmap (np.array): Heatmap generated by the model.
        ground_true (np.array): Ground truth mask.
    Returns:    
        IoU (list): List of IoU values for different thresholds.
    """
    IoU = []
    # Convert to numpy if using PyTorch tensors
    heatmap = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    ground_true = ground_true.numpy() if isinstance(ground_true, torch.Tensor) else ground_true
    
    # Ensure ground_true is a binary mask (0s and 1s)
    ground_true = ground_true.astype(bool)

    for thresh in np.linspace(0, 1, 36):
        # Convert heatmap to a binary mask based on the threshold
        bin_heatmap = heatmap > thresh
        
        # Convert to boolean arrays for bitwise operations
        bin_heatmap = bin_heatmap.astype(bool)
        
        # Intersection: pixels where both heatmap and ground truth are positive
        intersection = np.sum(bin_heatmap & ground_true)
        
        # Union: pixels where either heatmap or ground truth are positive
        union = np.sum(bin_heatmap | ground_true)
        
        # Avoid division by zero
        if union == 0:
            IoU.append(0.0)
        else:
            IoU.append(intersection / union)

    
    
    return IoU

def read_IoU(results_Iou:list):
    """Read the IoU results and calculate the mean IoU for each threshold.
    Args:
        results_Iou (list): List of IoU results for each image.
    Returns:
        mean_Iou (dict): Dictionary with the mean IoU for each threshold.
        list_Iou (list): List of mean IoU values for each threshold.
    """
    none_iou = 0
    mean_Iou = {}
    for tresh, i in enumerate(results_Iou[0]):
        mean_Iou[tresh] = 0
        
    for image_iou in results_Iou:
        if image_iou is not None:
            for tresh, Iou in enumerate(image_iou) :
                mean_Iou[tresh] += float(Iou)
        else:
            none_iou += 1
    
    list_Iou = []
    for tresh in mean_Iou:
        mean_Iou[tresh] /= (len(results_Iou) - none_iou)
        list_Iou.append(mean_Iou[tresh])
    return mean_Iou, list_Iou

from sklearn.metrics import accuracy_score, precision_score, f1_score
    
def get_best_Iou(result_Iou:list, best_loc:int=0):  
    """Get the best IoU for each image.
    Args:
        result_Iou (list): List of IoU results for each image.
        best_loc (int): Index of the best IoU to return.
    Returns:    
        best_Iou (list): List of best IoU values for each image.
    """
    best_Iou = []
    for Iou in result_Iou:
        best_Iou.append(float(Iou[best_loc]))
    return best_Iou


def to_save(fig:matplotlib.figure.Figure, name:str, exts:list=['pdf', 'png'], folder:str='figs'):
    """Save the figure in the specified formats.
    Args:   
        fig (matplotlib.figure.Figure): Figure to save.
        name (str): Name of the file to save.
        exts (list): List of extensions to save the figure in.
        folder (str): Folder to save the figure in.
    """
    for ext in exts:
        fig.savefig(f'{folder}/{name}.{ext}',  **opts_savefig)

# %pip install imageio[ffmpeg]

import imageio
def make_mp4(moviename, fnames, fps, codec='mpeg4', do_delete=True):
    # Create a video writer object
    writer = imageio.get_writer(moviename, fps=fps)  # Adjust the fps as needed

    # Write frames to the video
    for fname in fnames:
        img = imageio.imread(fname)
        writer.append_data(img)

    # Close the writer
    writer.close()

    if do_delete: 
        for fname in fnames: os.remove(fname)
    return moviename

def get_top_indices(tensor:torch.tensor, k:int=5):
    """
    Get the indices of the top k values in a tensor."""

    try : 
        return torch.topk(tensor, k=k)[1]
    except:
        return torch.topk(tensor, k=1)[1]   

def euclidean_distance(point_1, point_2):


    """
    Calculate the Euclidean distance between two points in a 2D plane.
    
    Parameters:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.
    
    Returns:
        float: The Euclidean distance between the two points.
    """
    x1, y1 = point_1
    x2, y2 = point_2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def twoD_Gaussian(x:np.array, y:np.array, xo:float, yo:float, sigma_x:float, sigma_y:float): #2D Gaussian function
    """2D Gaussian function.
    Args:   
        x (np.array): x-coordinates.
        y (np.array): y-coordinates.
        xo (float): x-coordinate of the center.
        yo (float): y-coordinate of the center.
        sigma_x (float): Standard deviation in the x direction.
        sigma_y (float): Standard deviation in the y direction.
    Returns:
        np.array: 2D Gaussian function values.
    """
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()


def to_heatmap(key_points:list, shape:tuple):  # a function use to apply 2d gaussian at designeted position (keypoints) on a image
    """Create a 2D Gaussian heatmap based on key points.
    Args:   
        key_points (list): List of key points (x, y) to place the Gaussian peaks.
        shape (tuple): Shape of the output heatmap (height, width).
    Returns:
        np.array: 2D Gaussian heatmap."""
    y, x = np.mgrid[0:shape[0], 0:shape[1]] # get x and y extents
    Gauss = np.zeros([shape[0]*shape[1]])
    for i in key_points:
        x0, y0 = i
        Gauss += twoD_Gaussian(x, y, x0, y0, .2*x.max(), .2*y.max())
    return Gauss


def collect_centered_map(args:dict, map:torch.tensor, position_prior:tuple):
    """Collect the centered map from the original map,
    rolling it to the position of interest.
    Args:
        args (dict): Arguments for the model.
        map (torch.tensor): Original map to be centered.
        position_prior (tuple): Position of the center of the map.  
    Returns:
        torch.tensor: Centered map."""
    diff_x, diff_y =  (args.resolution[0]//2)-position_prior[0] , (args.resolution[1]//2)-position_prior[1]
    map = torch.roll(map, [diff_x, diff_y], dims=[0, 1])
    #print(map)
    if diff_x < 0 :
        map[diff_x:, :] = torch.nan
    elif diff_x > 0:
        map[:diff_x, :] = torch.nan
    if diff_y < 0 :
        map[:, diff_y:] = torch.nan
    elif diff_y > 0:
        map[:, :diff_y] = torch.nan
    return(map)

import seaborn as sns
def display_heat(image:np.array, likelihood_map:torch.tensor, resolution:tuple, ax:matplotlib.axes.Axes):
    """Display the heatmap on top of the image.
    Args:
        image (np.array): Image to display.
        likelihood_map (torch.tensor): Heatmap to display.
        resolution (tuple): Resolution of the heatmap.
        ax (matplotlib.axes.Axes): Axes to display the image on.
    """
    import cv2

    shape_im = image.shape[:2]
    likelihood_map = np.array(likelihood_map).reshape(resolution)
    likelihood_map = cv2.resize(np.array(likelihood_map), (shape_im[1],shape_im[0]), interpolation=cv2.INTER_NEAREST)
    image_lin_display = cv2.resize(image, likelihood_map.T.shape, interpolation= cv2.INTER_LINEAR)
    sns.heatmap(likelihood_map, linewidth = 0 , annot = False, cmap=cmap, vmin=0, vmax=1, cbar = False, alpha=.5, ax=ax)
    ax.imshow(image_lin_display)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

def display_heat_two(image:np.array, likelihood_maps:torch.tensor, match:list, resolution:tuple, title_doc:str, save:bool=False):
    """Display the heatmap on top of the image.
        Args:
        image (np.array): Image to display.
        likelihood_map (torch.tensor): Heatmap to display.
        resolution (tuple): Resolution of the heatmap.
        ax (matplotlib.axes.Axes): Axes to display the image on.
        save (bool): Save the figure or not.
        title_doc (str): Name of the file to save.
    """ 
    import cv2

    fig, axs = plt.subplots(1, len(likelihood_maps), figsize=(10, 10))
    for likelihood_map, ax in zip(likelihood_maps, axs):
        shape_im = image.shape[:2]
        likelihood_map = likelihood_map[:,match] if type(match) is int else likelihood_map[:,match].sum(axis=1)
        likelihood_map = np.array(likelihood_map).reshape(resolution)
        likelihood_map = cv2.resize(np.array(likelihood_map), (shape_im[1],shape_im[0]), interpolation=cv2.INTER_NEAREST)
        image_lin_display = cv2.resize(image, likelihood_map.T.shape, interpolation= cv2.INTER_LINEAR)
        sns.heatmap(likelihood_map, linewidth = 0 , annot = False, cmap=cmap, vmin=0, vmax=1, cbar = False, alpha=.5, ax=ax)
        ax.imshow(image_lin_display)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

    if save :
        to_save(fig, name=title_doc)
