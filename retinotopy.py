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
tic = time.time()
from time import strftime, gmtime
datetag = strftime("%Y-%m-%d", gmtime())
#datetag = '2024-05-24'
# datetag = '2025-01-05'
datetag = '2025-03-06' # Jean Zay

import cv2
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
# import seaborn as sns
# import sklearn.metrics

plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

# matplotlib parameters
from matplotlib import font_manager
# Fig variables
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue",  "lightsteelblue", "lavender", "white", "seashell", "mistyrose",  "firebrick"])

fig_width = 15
fontsize = 14
font = font_manager.FontProperties(weight='normal', size=fontsize)
dpi = 'figure'
dpi = 200
opts_savefig = dict(dpi=dpi, bbox_inches='tight', pad_inches=0, edgecolor=None)

colors = ['b', 'r', 'k', 'g', 'm', 'y']
fig_width = 20
phi = (np.sqrt(5)+1)/2 # golden ratio for the figures :-)

#to plot & display 
def pprint(message): #display function
    print('-'*len(message))
    print(message)
    print('-'*len(message))

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
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

def get_filename(data_cache, datetag, data_set_type, model_name, do_polar):
    # return f'{data_cache}/{datetag}_{data_set_type}_{model_name}_{do_polar=}'
    return f"{data_cache}/{datetag}_{data_set_type}_{model_name}_{'retino' if do_polar else 'cartesian'}"

exts = ['pdf', 'svg', 'png']
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
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU : ', torch.cuda.get_device_name(), '#GPU=', torch.cuda.device_count())    
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# set seed function
def set_seed(seed=None, seed_torch=True, verbose=False):
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
#############################################################

#############################################################
data_cache = 'cached_data'
os.makedirs(data_cache, exist_ok=True)

interpolation = T.InterpolationMode.BILINEAR
padding_mode = "border"

batch_size = 250

USER = os.environ['USER']  # username

if USER=='uvb28bo': # Jean Zay
    # DATAROOT = '../data'
    DATAROOT = f'{os.environ['SCRATCH']}/data'
    num_workers = 8
    batch_size = 512
    print(f'Running on Jean Zay with {torch.cuda.get_device_name()} with {DATAROOT=} and {USER=} ')
elif '.cluster' in HOST: # mesocentre
    DATAROOT = '/scratch/lperrinet/science/Deep_learning/data'
    num_workers = 8
elif 'm-gpu' in HOST: 
    DATAROOT = 'data'
    # batch_size = 50
    num_workers = 2
    batch_size = 512
elif HOST in ['babbage']: # 
    DATAROOT = '/data/Deep_learning/data'
    num_workers = 2
elif HOST in ['CONECT-LID-01']: # emmy
    # DATAROOT = '/envau/userspace/perrinet.l/data'
    DATAROOT = 'data'
    batch_size = 50
    num_workers = 16    
elif HOST in ['CONEC-LID-002']: # faraday
    # DATAROOT = '/envau/userspace/perrinet.l/data'
    DATAROOT = '/scratch/ImageNet'
    num_workers = 16    
elif HOST in ['inv-ope-de06', 'INV-133-DE01']: # CURIE , ada
    DATAROOT = '/data/JNJER/Deep_learning/data'
    num_workers = 2
elif HOST in ['neo-ope-de04']: # Darwin  
    DATAROOT = '/data/JNJER/Deep_learning/data'
    num_workers = 16
elif HOST in ['brain-lid-004']: # GPU manu  
    DATAROOT = '/data/JNJER/Deep_learning/data'
    num_workers = 16
elif 'obiwan' in HOST: 
    # DATAROOT = '/Volumes/UnaTera/2023_archives/2023_science/JNJER_PhD/data'
    DATAROOT = '/Volumes/data/2024_archives/2024_science/Deep_learning/data'
    DATAROOT = '/Volumes/SSD1TO/ImageNet'
    DATAROOT = '/Volumes/SSD1TO/Deep_learning/ILSVRC2010_ImageNet'
    DATAROOT = 'data'
    DATAROOT = '/Volumes/SSD1TO/Deep_learning/data'
    interpolation = T.InterpolationMode.NEAREST
    padding_mode = "reflection"
    num_workers = 2
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
    
    image_size: int = 224 #
    num_epochs: int = 20 # 
    n_train_stop: int = 0 # set to zero to use all images
    seed: int = 1998 # Set the seed for reproducibility 
    batch_size: int = batch_size # Set number of images per input batch
    batch_size_val: int = batch_size # Set number of images per input batch
    lr: float = 5.e-5 # Set learning rate 
    momentum: float = .02 # Set the momentum
    beta2: float = 0 # Set the second momentum - use SGD if set to 0
    rs_min: float = 0.00
    rs_max: float = -5.00
    
    do_polar: bool = True # use a retinotopic mapping
    do_raw: bool = False
    do_translate: bool = False
    do_resize: bool = True # resize the image to args.image_size
    do_mask: bool = True # add a circular mask on the Cartesian input to match the retino input (circular window) 
    do_scratch: bool = True # whether we use pretrained weights or not during transfer learning
    do_rotation: bool = False # just use this for rotation attacks
    # todo remove as it is not used 
    do_rot_train: bool = False # just use this for training with rotation 
    resolution: tuple = (11, 11) # resolution of the likelihood map
    size_ratio: float = 0.1 # how much of the image to use relative to radius
    do_saccade: bool = False # True to get multiple pov for eah image in the data set transform
    do_zoom: bool = False # True to apply a zoom (range from args.size_ratio to (2 + args.size_ratio) +/- 0.1 )
    method: str = 'valid' #select sampling for mapping between full = with border & valid = no border
    saccade_type: str = 'multi' #select sampling for mapping between multi = with multiple ratio & grid = same sample ratio 
    angles = np.linspace(-180, 180, 100)   # combination of angles used for training or attacks
    normalize: bool = True
    
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
# import nltk
# nltk.download('wordnet')

with open(args.loader) as json_file:
    Imagenet_urls_ILSVRC_2016 = json.load(json_file)
    
match = {}
labels = []
revlabels_dico = {}
labels_dico = {}
i_labels_dico = {}

for task in args.tasks:
    match[task] = []
    
#----------------Get the label for the Imagenet categorization------------------------

for i_img, img_id in enumerate(Imagenet_urls_ILSVRC_2016):
    syn_= wn.synset_from_pos_and_offset('n', int(img_id.replace('n','')))
    sem_ = syn_.hypernym_paths()[0]
    label = syn_.name().split(".")[0]
    labels.append(label)
    i_labels_dico[label] = i_img
    labels_dico[label] = img_id
    revlabels_dico[img_id] = label
    for i in np.arange(len(sem_)):
        for task in args.tasks:
            if sem_[i].lemmas()[0].name() in task :
                match[task].append(i_img)

#---------------Get annotations from other the data set (Animal10k, ...)---------------
def get_annotation(type):

    # annotations_animal: str = f'{DATAROOT}/Animal10k_annotations.json' # File containing Animak10k's labels
    # annotations_val: str = f'{DATAROOT}/LOC_val_solution_with_sizes.csv' # File containing Imagenets's labels

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

def imgs_to_np(img_list, im_mean=im_mean, im_std=im_std):
    images = torchvision.utils.make_grid(img_list, nrow=11)
    """Imshow for Tensor."""
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)

def imshow(img_list, im_mean=im_mean, im_std=im_std, 
           title=None, fig_height=7, fig=None, ax=None, save=False, name=None): #allow to display the input image
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
    
def to_dev(args, item):
    if hasattr(args, 'device'):
        return item.to(args.device)
    else:
        return item

def get_grid(args, endpoint=False):

    rs_ = torch.logspace(args.rs_min, args.rs_max, args.image_size, base=2) 
    if endpoint:
        ts_ = torch.linspace(0, torch.pi*2, args.image_size)
    else:
        ts_ = torch.linspace(0, torch.pi*2, args.image_size+1)[:-1]     
    grid_xs = torch.outer(rs_, torch.cos(ts_)) 
    grid_ys = torch.outer(rs_, torch.sin(ts_))
    
    return to_dev(args, torch.stack((grid_xs, grid_ys), 2))

def get_zoom_grid(args):
    grids = []
    for ratio in np.arange(args.size_ratio, (10 + args.size_ratio), 0.1):
        if args.do_polar:
            start = get_start(ratio)
            rs_ = torch.logspace(start, args.rs_max, args.image_size, base = 2)
            ts_ = torch.linspace(0, torch.pi*2, args.image_size+1)[:-1]  
        
            grid_x = torch.outer(rs_, torch.cos(ts_)) 
            grid_y = torch.outer(rs_, torch.sin(ts_)) 
            
        else:
            x = torch.linspace(-ratio, ratio, args.image_size)
            y = torch.linspace(-ratio, ratio, args.image_size)
            grid_y, grid_x = torch.meshgrid(x, y, indexing='ij')

        
        grids.append(torch.stack((grid_x, grid_y), 2))
        
    return torch.stack(grids)

def generate_translate_roll_grids(args):
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


def get_start(start_value):
    return torch.log2(torch.tensor(start_value)).item()

def get_saccade_map(args):
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
    return num//denum + num%denum


def multi_sacade_map(args):
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

def apply_grid(image, grid): 
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

    
def make_mask(image_size, radius = 0.5):
    X, Y = np.meshgrid(np.linspace(-radius, radius, image_size, endpoint=True), 
               np.linspace(-radius, radius, image_size, endpoint=True))
    R = np.sqrt(X**2 + Y**2)
    mask = (R < 0.5).astype(np.float32)
    return torch.from_numpy(mask)

class ApplyMask: 
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, images):
        return images[:, :, ::] * self.mask

class CleanRotations_class(object): 
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
def get_transforms(args, im_mean=im_mean, im_std=im_std):
    
    transforms = [                
        T.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        T.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    ]   

    if args.do_rot_train: # used for augmentation and testing rotations
        transforms.append(T.RandomRotation(degrees=(min(args.angles), max(args.angles)), interpolation=interpolation, expand=False))

    if args.do_rotation and not args.do_saccade:
        args.batch_size_val, args.batch_size = 1, 1
        grid = get_grid(args).repeat(len(args.angles), 1, 1, 1)
        transforms.append(CleanRotations_class(args.angles))

    if args.do_zoom and not (args.do_saccade or args.do_translate):
        args.batch_size_val, args.batch_size = 1, 1
        grid_zoom = to_dev(args, get_zoom_grid(args))
        transforms.append(transform_apply_grid(grid_zoom, 'multiple'))
        
        if not args.do_polar:
            mask = to_dev(args, make_mask(args.image_size))
            transforms.append(ApplyMask(mask))

    if args.do_translate and not (args.do_saccade or args.do_zoom):
        args.batch_size_val, args.batch_size = 1, 1
        grid_translate = to_dev(args, generate_translate_roll_grids(args))
        transforms.append(transform_apply_grid(grid_translate, 'multiple'))
     

    if args.do_polar and not (args.do_saccade or args.do_zoom or args.do_translate):
        grid_polar = get_grid(args) if not args.do_rotation else get_grid(args).repeat(len(args.angles), 1, 1, 1)
        transforms.append(transform_apply_grid(grid_polar, ('base' if not args.do_rotation else None)))

    if args.do_resize and not (args.do_polar or args.do_saccade or args.do_zoom):
        transforms.append(T.Resize(int(args.image_size), interpolation=interpolation, antialias=True))
        transforms.append(T.CenterCrop((int(args.image_size), int(args.image_size))))
        
    
    if args.do_mask and not (args.do_polar or args.do_saccade or args.do_zoom):
        mask = to_dev(args, make_mask(args.image_size))
        transforms.append(ApplyMask(mask))

    if args.do_saccade :
        args.batch_size_val = 1
        grid = to_dev(args, multi_sacade_map(args)) if args.saccade_type == 'multi' else to_dev(args, get_saccade_map(args)) 
        transforms.append(transform_apply_grid(grid, 'multiple'))

        if not args.do_polar and not args.do_raw:
            mask = to_dev(args, make_mask(args.image_size))
            transforms.append(ApplyMask(mask))

    if args.normalize:
        transforms.append(T.Normalize(mean=im_mean, std=im_std)) # to normalize colors on the imagenet dataset
    
    return T.Compose(transforms)

from torchvision.datasets import ImageFolder

def is_valid_file(path):
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


def image_datasets_transforms(args, im_mean=im_mean, im_std=im_std, verbose=True):

    image_datasets  = {}
    for folder in args.folders:

        data_transform = get_transforms(args, im_mean=im_mean, im_std=im_std)

        # load the data
        path = os.path.join(args.root, folder) # data path
        image_datasets[folder] = ImageFolder(path, 
                                             transform=data_transform,
                                             is_valid_file=is_valid_file)

        if verbose: 
            print(f"Loaded {len(image_datasets[folder])} images under {folder}")  

    return image_datasets # bug ? on renvoie que le dernier dataset de args.folder


def datasets_transforms(args, im_mean=im_mean, im_std=im_std,
                        num_workers=num_workers, pin_memory=True, shuffle=True, verbose=True):
    """
    
    quel rapport avec image_datasets_transforms ?

    
    TODO: obsolete = "if angle is not none, applies a random rotation"
    """


    image_datasets  = image_datasets_transforms(args, im_mean=im_mean, im_std=im_std, verbose=verbose)

    dataloaders = {}
    for folder in args.folders:

        dataloaders[folder] = torch.utils.data.DataLoader(
                                image_datasets[folder], 
                                batch_size=args.batch_size if folder=='train' else args.batch_size_val,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
                        )
        # if verbose: 
        #     print(f"Loaded {len(image_datasets)} images under {folder}")  

    return dataloaders


#############################################################

#############################################################
def make_padding_circular_again(model_retrain):
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

def train_model(args, model, dataloaders, each_steps=64, verbose=True):
    
    # retraining the full model
    for param in model.parameters():
        param.requires_grad = True        

    # sets the optimizer
    if args.beta2 > 0.: 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(1-args.momentum, 1-args.beta2)) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=1-args.momentum) # to set training variables
    
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
    criterion = nn.CrossEntropyLoss() # binary_cross_entropy_with_logits

    # the DataFrame to record from
    df_train = pd.DataFrame([], columns=['epoch', 'i_image', 'total_image', 'avg_loss', 'avg_acc', 'avg_loss_val', 'avg_acc_val', 'time']) 


    since = time.time()
    total_image = 0
    n_train = len(dataloaders['train'].dataset)
    n_train_stop = args.n_train_stop
    if n_train_stop==0: n_train_stop = n_train

    for i_epoch in range(args.num_epochs):
        i_image = 0
        for i_step, (images, labels) in enumerate(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)
            total_image += len(images)
            i_image += len(images)
            if i_image > n_train_stop: break # early stopping

            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            outputs = model(images)
             
            loss = criterion(outputs, labels)            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, dim=1)

            avg_loss = loss.item() * images.size(0)
            avg_acc = torch.mean((preds == labels.data)*1.).cpu().item()

            if (i_step % (max(n_train_stop//args.batch_size//each_steps, 1))==0) or (i_step == n_train_stop-1):
                with torch.no_grad():
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
                    if verbose:  print(f"Epoch {i_epoch}, i_image {i_image} : train= loss: {avg_loss:.4f} / acc : {avg_acc:.4f} - val= loss : {avg_loss_val:.4f} / acc : {avg_acc_val:.4f} / time:{time.time() - since:.1f}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()        
    return model, df_train

import torchvision.models as models

models_url = {'resnet18': url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
              'resnet50': url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
              'resnet101': url="https://download.pytorch.org/models/resnet101-cd907fc2.pth"}

def apply_weights(model_name):
    model_path = os.path.join(data_cache, f'{model_name}.pth')
    
    if not os.path.exists(model_path):
        model = models.__dict__[model_name](pretrained=True)
        torch.save(model.state_dict(), model_path)
    else:
        model = models.__dict__[model_name]()
        model.load_state_dict(torch.load(model_path), map_location=torch.device(device), weights_only=True)
    
    return model

def load_model(model_name='resnet50', model_path=None, do_scratch=False, do_circular=False, verbose=True):
    # get the architecture of the network
            
    if model_name=='resnet18':
        # model = torchvision.models.resnet18(weights=None if do_scratch else torchvision.models.ResNet18_Weights.DEFAULT)
        model = torchvision.models.resnet18(weights=None)
    elif model_name=='resnet50':
        # model = torchvision.models.resnet50(weights=None if do_scratch else torchvision.models.ResNet50_Weights.DEFAULT)
        model = torchvision.models.resnet50(weights=None)
    elif model_name=='resnet101':
        # model = torchvision.models.resnet101(weights=None if do_scratch else torchvision.models.ResNet101_Weights.DEFAULT)
        model = torchvision.models.resnet101(weights=None)
    else:
        raise ValueError(f'Unknown model {model_name}')
    
    if not(model_path is None):
        if verbose: print(f'loading .... {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))

    if do_circular:
        model = make_padding_circular_again(model)

    return model
#############################################################


#############################################################

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

def compute_likelihood_map(args, model, image, resolution=(11, 11), # how many fixation points to use
                           size_ratio=args.size_ratio, # how much of the image to use relative to radius
                           N_batch=125):

    pos_H, pos_W, box_size = get_positions(args, image)
    args.device = device
    data_transform = get_transforms(args)
    # image = image.to(device)
    # model = model.to(device)

    N_fixations = resolution[0] * resolution[1]
    proba_label = torch.zeros((N_fixations, 1000))
    for idx_start in np.arange(0, N_fixations, N_batch):
        idx_stop = np.min((idx_start+N_batch, N_fixations))
            
        with torch.no_grad():
            cropped_images = torch.empty((idx_stop-idx_start, 3, box_size, box_size), device=device)

            for i_fixation, (h, w) in enumerate(zip(pos_H.ravel()[idx_start:idx_stop], 
                                                    pos_W.ravel()[idx_start:idx_stop])):
                h, w = int(h), int(w)
                cropped_image = crop(image, h-box_size//2, w-box_size//2, box_size, box_size)
                cropped_image = cropped_image.to(device)
                cropped_images[i_fixation, ...] = data_transform(cropped_image)
            print(cropped_images.shape)
            cropped_images = T.Resize((224, 224), interpolation=interpolation, antialias=True)(cropped_images)
            print(cropped_images.shape)
            outputs = torch.nn.functional.softmax(model(cropped_images), dim=1)
        
        proba_label[idx_start:idx_stop, :] = outputs#.detach().cpu().numpy()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    return proba_label
    #return pos_H, pos_W, proba_label

def clean_resize(args, image):
    image = T.Resize(args.image_size, interpolation=interpolation, antialias=True)(image)
    return T.CenterCrop((int(args.image_size), int(args.image_size)))(image)
           

def get_batch(args, model, full_image, size=100):
    N_fixations = args.resolution[0] * args.resolution[1]
    proba_label = torch.zeros((N_fixations, 1000))
    for idx_start in np.arange(0, N_fixations, size):
        idx_stop = np.min((idx_start+size, N_fixations))
        with torch.no_grad():
            outputs = torch.nn.functional.softmax(model(full_image[idx_start:idx_stop]), dim=1)
        proba_label[idx_start:idx_stop, :] = outputs#.detach().cpu().numpy()
    return proba_label

def rolling_map(args, image, retino_grid, set_resolution=args.resolution, subsample_size=None):
    if subsample_size is None:
        #subsample_size = args.image_size
        subsample_size = min(image.shape[1], image.shape[2]) * args.size_ratio
    preds_im = []
    for i in np.linspace(0, (image.shape[1]-subsample_size), set_resolution[0], dtype=int):
        for j in np.linspace(0, (image.shape[2]-subsample_size), set_resolution[1], dtype=int):
            if args.do_polar:
                preds_im.append(apply_grid(image[:,i:(subsample_size+i),j:(subsample_size+j)].unsqueeze(0), retino_grid))
            else:
                #preds_im.append(clean_resize(args, image[:,i:(subsample_size+i),j:(subsample_size+j)]))
                preds_im.append(T.Resize((int(args.image_size), int(args.image_size)),
                    interpolation=interpolation, antialias=True)(image[:,i:(subsample_size+i),j:(subsample_size+j)]))
    return torch.stack(preds_im).reshape(set_resolution[0],set_resolution[1], 3, args.image_size, args.image_size).squeeze(0)

def rolling_map_LP(args, origin_size, image, retino_grid):
    all_image_size = np.linspace(min(origin_size), args.image_size, round(args.resolution[0]/2), dtype=int)
    grids_images = []
    for num, i in enumerate(np.linspace(1, args.resolution[0], round(args.resolution[0]/2), dtype=int)):
        resolution_grid = (i,i)
        subsample_size = all_image_size[num]
        grids_images.append(rolling_map(args, image, retino_grid, resolution_grid, subsample_size))
        
    if args.do_polar:
        grids_images[0] = apply_grid(image.unsqueeze(0), retino_grid)
    else:
        #grids_images[0] = clean_resize(args, image)
        grids_images[0] = T.Resize((int(args.image_size), int(args.image_size)), interpolation=interpolation, antialias=True)(image)
    for i in np.linspace(0, len(grids_images)-2, len(grids_images)-1, dtype=int):
        grids_images[i+1][1:-1,1:-1] = grids_images[i]
        
    grids_images = grids_images[i+1][:, :, ::].reshape(args.resolution[0] * args.resolution[1], 3,
                                                                args.image_size, args.image_size)
    if args.do_polar:
        return grids_images
    else:
        return grids_images * to_dev(args, make_mask(args.image_size))

#############################################################

def to_tuple(str_size):
    return (int(str_size.split(' ')[0].split('(')[1].split(',')[0]),
            int(str_size.split(' ')[1].split(')')[0]))

def get_boxes_imagenet(box_anot): # function to get a list of dict for each box
    
    box = []
    for i in np.arange(len(box_anot)//5) :  # iterate for each box
        box.append({'ymin' : int(box_anot[(i*5)+1]),
                     'xmin' : int(box_anot[(i*5)+2]),
                     'ymax' : int(box_anot[(i*5)+3]),
                     'xmax' : int(box_anot[(i*5)+4])})
    return box

def get_mask_from_bb(target_size, orig_size, boxes):
    
    mask = np.zeros((orig_size),dtype=np.uint8) # initialize mask
    for box in boxes : 
        mask[box['ymin']:box['ymax'],box['xmin']:box['xmax']] = 1 # fill with white pixels
    return cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR).T

def store_pandas(df, df_):
    if df is None:
        return df_
    else:
        return(pd.concat([df, df_], ignore_index=True))

def normalize_array(arr):
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


def little_box(boxe, origin_size, resolution):
    x_mid = (boxe['xmax'] - boxe['xmin']) + boxe['xmin']
    y_mid = (boxe['ymax'] - boxe['ymin']) + boxe['ymin']
    return (int(np.round((x_mid*resolution[0])/origin_size[0])-1), int(np.round((y_mid*resolution[1])/origin_size[1]))-1)
    

def no_axis_title(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{title}')
    return 

def get_three_points(ground_true_indices, resolution):

    
    X_mid = math.ceil(np.mean(ground_true_indices[0]%min(args.resolution)))
    Y_mid = math.ceil(np.mean(ground_true_indices[0]//min(args.resolution)))
    
    if X_mid < ((len(resolution)//2)+1) : 
        X_bord = max(ground_true_indices[0]%resolution[0])
    else :
        X_bord = min(ground_true_indices[0]%resolution[0])
    
    if Y_mid < ((len(resolution)//2)+1) : 
        
        Y_bord = max(ground_true_indices[0]//resolution[1])
    else :
        Y_bord = min(ground_true_indices[0]//resolution[1])
    
    if X_mid < ((len(resolution)//2)+1) :
        if Y_mid < ((len(resolution)//2)+1) :
            end = (len(resolution),len(resolution))
        else:
            end = (0,len(resolution))
    else:
        if Y_mid < ((len(resolution)//2)+1) :
            end = (len(resolution),0)
        else:
            end = (0,0)

    return [(X_mid , Y_mid), (X_bord , Y_bord), end]

def get_points_between(p1, p2):
    points = []
    x1, y1 = p1
    x2, y2 = p2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    
    err = dx - dy
    
    x, y = x1, y1
    first_step = True
    
    while (x, y) != (x2, y2):
        if not first_step:  # Skip adding the starting point
            points.append((x, y))
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        
        first_step = False
    
    # Ensure at least one point is returned and is not equal to the start or end point

    if len(points) == 0 :
        mid_x = (x1 + x2) // 2
        mid_y = round_up((y1 + y2), 2)
        points = [(mid_x, mid_y)]
    
    return points



def get_like_point(heatmap, resolution, three_points):
    mid_point = heatmap[(three_points[0][0]*resolution[0]) + three_points[0][1]]
    in_point = heatmap[(three_points[1][0]*resolution[0]) + three_points[1][1]]
    ext_point = heatmap[(three_points[2][0]*resolution[0]) + three_points[2][1]]
    return mid_point, in_point, ext_point


def get_ground_true(args, image_name, annotations, mode):
    
    if mode == 'Imagenet':
        box_anot = annotations[annotations['ImageId'] == image_name]['PredictionString'].item().split(' ')
        boxes = get_boxes_imagenet(box_anot)
        origin_size = to_tuple(annotations[annotations['ImageId'] == image_name]['origin_size'].item())
        ground_true = np.round(get_mask_from_bb(args.resolution, origin_size, boxes))
    
    else: #Get ground true for Animal 10k
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
                return None, None, np.ones(1), origin_size # return None for value we cant caculate and 1 to stop at the next condition
            
    ground_true_indices = np.where(ground_true.reshape(args.resolution[0]*args.resolution[1]) > 0)
    
    three_points = get_three_points(ground_true_indices, args.resolution)
    
    return ground_true_indices, three_points, ground_true, origin_size

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def get_IoU(heatmap, ground_true):
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

def read_IoU(results_Iou):
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
    
def get_best_Iou(result_Iou, best_loc):
    best_Iou = []
    for Iou in result_Iou:
        best_Iou.append(float(Iou[best_loc]))
    return best_Iou

def get_dist(results_dist):
    
    distances = {0:[], 1:[]}
    for dist in results_dist:
        for pos, dist_ in enumerate(dist.split(' ')) :
            distances[pos].append((float(dist_.replace(',', '').replace('[', '').replace(']', ''))))
    return [np.zeros(len(distances[0])), np.array(distances[0]), np.array(distances[1])]

def to_save(fig, name, exts=['pdf', 'png'], folder='figs'):
    for ext in exts:
        fig.savefig(f'{folder}/{name}.{ext}',  **opts_savefig)

# https://moviepy.readthedocs.io/en/latest/getting_started/videoclips.html#imagesequenceclip
# def make_mp4(mp4name, fnames, fps, do_delete=True):
#     from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
#     clip = ImageSequenceClip(fnames, fps=fps)
#     clip.write_videofile(mp4name, fps=fps, codec='libx264', verbose=False, logger=None)
#     if do_delete: 
#         for fname in fnames: os.remove(fname)
#     return mp4name

def make_mp4(mp4name, fnames, fps, codec='mpeg4', do_delete=True):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    # from moviepy import ImageSequenceClip
    clip = ImageSequenceClip(fnames, fps=fps)
    clip.write_videofile(mp4name, codec=codec, fps=fps, logger=None)
    if do_delete: 
        for fname in fnames: os.remove(fname)
    return mp4name


def get_top_indices(tensor, k=5):

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


def twoD_Gaussian(x, y, xo, yo, sigma_x, sigma_y): #2D Gaussian function
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()


def to_heatmap(key_points, shape):  # a function use to apply 2d gaussian at designeted position (keypoints) on a image
    y, x = np.mgrid[0:shape[0], 0:shape[1]] # get x and y extents
    Gauss = np.zeros([shape[0]*shape[1]])
    for i in key_points:
        x0, y0 = i
        Gauss += twoD_Gaussian(x, y, x0, y0, .2*x.max(), .2*y.max())
    return Gauss


def collect_centered_map(args, map, position_prior):
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
def display_heat(image, likelihood_map, resolution, model_name, ax):
    shape_im = image.shape[:2]
    likelihood_map = np.array(likelihood_map).reshape(resolution)
    likelihood_map = cv2.resize(np.array(likelihood_map), (shape_im[1],shape_im[0]), interpolation=cv2.INTER_NEAREST)
    image_lin_display = cv2.resize(image, likelihood_map.T.shape, interpolation= cv2.INTER_LINEAR)
    sns.heatmap(likelihood_map, linewidth = 0 , annot = False, cmap=cmap, vmin=0, vmax=1, cbar = False, alpha=.5, ax=ax)
    ax.imshow(image_lin_display)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

def display_heat_two(image, likelihood_maps, match, resolution, title_doc, save=False):
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
