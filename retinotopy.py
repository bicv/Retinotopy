from time import strftime, gmtime
datetag = strftime("%Y-%m-%d", gmtime())
datetag = '2024-04-25'

# MATPLOTLIB imports and parameters
import numpy as np
import json
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
cmap = plt.cm.get_cmap('viridis')
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

exts = ['pdf', 'svg', 'png']

# Wordnet
from nltk.corpus import wordnet as wn
#from numpy import random
import os
import requests
import time

# to store results
import pandas as pd

# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass, asdict, field

import os
HOST = os.uname()[1]
print(f'{HOST=}')

def touch(fname): open(fname, 'w').close()

# Importing libraries
import torch
import torch.nn.functional as nnf
import torchvision
# from torchvision import datasets, models, transforms
# from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
import torch.nn as nn
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

data_cache = 'cached_data'
interpolation = T.InterpolationMode.BILINEAR
batch_size = 50
if '.cluster' in HOST: # mesocentre
    DATAROOT = '/scratch/lperrinet/science/Deep_learning/data'
    num_workers = 1
elif HOST in ['babbage']: # 
    DATAROOT = '/data/Deep_learning/data'
    num_workers = 2
elif HOST in ['CONEC-LID-001']: # emmy
    DATAROOT = '/data/JNJER/Deep_learning/data'
    num_workers = 16
elif HOST in ['CONEC-LID-002']: # faraday
    # DATAROOT = '/envau/userspace/perrinet.l/data'
    DATAROOT = '/scratch'
    num_workers = 16    
elif HOST in ['inv-ope-de06', 'INV-133-DE01']: # CURIE , ada
    DATAROOT = '/data/JNJER/Deep_learning/data'
    num_workers = 2
elif HOST in ['neo-ope-de04']: # Darwin
    DATAROOT = '/data/JNJER/Deep_learning/data'
    num_workers = 16
elif 'obiwan' in HOST: 
    # DATAROOT = '/Volumes/UnaTera/2023_archives/2023_science/JNJER_PhD/data'
    DATAROOT = '/Volumes/SSD1TO/ImageNet'
    interpolation = T.InterpolationMode.NEAREST
    num_workers = 4
elif 'Ahsoka' in HOST: 
    DATAROOT = '/Volumes/data/2024_archives/2024_science/Deep_learning/data'
    DATAROOT = '/Volumes/backups/2023_archives/2023_science/JNJER_PhD/data'
    num_workers = 4
elif 'DESKTOP-27VNO0E' in HOST: 
    DATAROOT = '/mnt/d/Data/'
    num_workers = 16
else:
    DATAROOT = data_cache
    num_workers = 1


if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Running on metal', device)
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU : ', torch.cuda.get_device_name(), '#GPU=', torch.cuda.device_count())
else:
    device = torch.device('cpu')

# device = torch.device('cpu')
# torch.__version__, device

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


def charge_model(model_name='resnet50', model_path=None, do_scratch=False, do_polar=False):
    # get the architecture of the network
            
    if model_name=='resnet18':
        model = torchvision.models.resnet18(weights=None if do_scratch else torchvision.models.ResNet18_Weights.DEFAULT)
    elif model_name=='resnet50':
        model = torchvision.models.resnet50(weights=None if do_scratch else torchvision.models.ResNet50_Weights.DEFAULT)
    elif model_name=='resnet101':
        model = torchvision.models.resnet101(weights=None if do_scratch else torchvision.models.ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f'Unknown model {model_name}')
    
    if not(model_path is None):
        print(f'loading .... {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    if do_polar:
        model = make_padding_circular_again(model)

    return model


# data_set_type = 'focus' # Select your root between : 'boxes', 'focus', 'full'

@dataclass
class Params:
    import platform
    print('Welcome on', platform.platform())

    datetag: str = datetag # Set the date of the result's file
    loader: str = f'{DATAROOT}/Imagenet_urls_ILSVRC_2016.json' # File containing Imagenet's labels
    # annotations: str = f'{DATAROOT}/animal_10k/ap-10k/annotations/clean_annotations.json' # File containing Imagenet's labels

    # root: str = f'{DATAROOT}/Imagenet_{data_set_type}' # Directory containing images to perform the training
    folders: list = field(default_factory=lambda: ['val', 'train']) # Set the training and validation folders relative to the root
    
    image_size: int = 224 #
    num_epochs: int = 1 # 5 # 
    n_train_stop: int = 0 # set to zero to use all images
    seed: int = 1998 # Set the seed for reproducibility 
    batch_size: int = batch_size # Set number of images per input batch
    batch_size_val: int = batch_size # Set number of images per input batch
    lr: float = 0.00015 # Set learning rate 
    momentum: float = .06 # Set the momentum
    beta2: float = 0 # S
    rs_min: float = 0.05
    rs_max: float = -4.95
    do_polar: bool = True
    do_scratch: bool = True # whether we use pretrained weights or not during transfer learning
    do_rotation: bool = False
    
    torch.manual_seed(seed)
    
args = Params()

# keep a human readable copy of the parameters
json_fname = os.path.join(data_cache, datetag + '_config_args.json')
with open(json_fname, 'wt') as f:
    json.dump(vars(args), f, indent=4)


#DCCN training
# annotations = json.load(open(args.annotations) ) 

with open(args.loader) as json_file:
    Imagenet_urls_ILSVRC_2016 = json.load(json_file)
match = []

# import nltk
# nltk.download('wordnet')
#----------------Get the label for the Imagenet categorization------------------------
for i_img, img_id in enumerate(Imagenet_urls_ILSVRC_2016):
    syn_= wn.synset_from_pos_and_offset('n', int(img_id.replace('n','')))
    sem_ = syn_.hypernym_paths()[0]
    for i in np.arange(len(sem_)):
        if sem_[i].lemmas()[0].name() in 'animal' :
            match.append(i_img)
#------------------------------------------------------------------------------------

im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225])

def imgs_to_np(img_list):
    images = torchvision.utils.make_grid(img_list)
    """Imshow for Tensor."""
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)

def imshow(img_list, title=None, fig_height=5): #allow to display the input image
    fig = plt.figure(figsize=(fig_height*len(img_list), fig_height))
    inp = imgs_to_np(img_list)
    plt.imshow(inp)
    plt.xticks([]) ; plt.yticks([])
    if title is not None: plt.title(title)
    plt.tight_layout()
    fig.set_facecolor(color='white')
    plt.show()


print(f'On date {args.datetag}, Running learning on host {HOST} with device {device}')

# n_r, n_t = args.image_size, args.image_size 
# n_r, n_t = args.image_size, 256 
    
match = []
labels = []
#----------------Get the label for the Imagenet categorization------------------------
for i_img, img_id in enumerate(Imagenet_urls_ILSVRC_2016):
    syn_= wn.synset_from_pos_and_offset('n', int(img_id.replace('n','')))
    sem_ = syn_.hypernym_paths()[0]
    labels.append(syn_.name().split(".")[0])
    for i in np.arange(len(sem_)):
        if sem_[i].lemmas()[0].name() in 'animal' :
            match.append(i_img)


def get_grid(args, endpoint=False):

    rs_ = torch.logspace(args.rs_min, args.rs_max, args.image_size, base=2) 
    if endpoint:
        ts_ = torch.linspace(0, torch.pi*2, args.image_size)
    else:
        ts_ = torch.linspace(0, torch.pi*2, args.image_size+1)[:-1]     
    grid_xs = torch.outer(rs_, torch.cos(ts_)) 
    grid_ys = torch.outer(rs_, torch.sin(ts_))

    return torch.stack((grid_xs, grid_ys), 2)


# def to_retino_tens(images, grid): 
#     grid = grid.repeat(images.shape[0],1,1,1)
#     return nnf.grid_sample(images, grid, 
#                            padding_mode="border", align_corners=False).squeeze(dim=0)



class to_log_polar_tens(object): 
    def __init__(self, logPolar_grid):
        self.grid = logPolar_grid

    def __call__(self, images):
        return nnf.grid_sample(images.unsqueeze(0), self.grid.unsqueeze(0), 
                               padding_mode="border", align_corners=False).squeeze(dim=0)
    
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
    

# Resnet 101 datasets initialisation
def get_transforms(args, im_mean=im_mean, im_std=im_std, angle_min=-180, angle_max=180):

    grid = get_grid(args)
    mask = make_mask(args.image_size)

    transforms = [                
        T.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        #T.ToImageTensor(),
        T.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    ]   
    if args.do_rotation:
        transforms.append(T.RandomRotation(degrees=(angle_min, angle_max), interpolation=interpolation, expand=False))
    if args.do_polar: 
        transforms.append(to_log_polar_tens(grid))
    else:
        transforms.append(T.Resize(int(args.image_size), interpolation=interpolation, antialias=True))
        transforms.append(T.CenterCrop((int(args.image_size), int(args.image_size))))
        transforms.append(ApplyMask(mask))

    transforms.append(T.Normalize(mean=im_mean, std=im_std)) # to normalize colors on the imagenet dataset
    
    return T.Compose(transforms)

def datasets_transforms(args, im_mean=im_mean, im_std=im_std, angle_min=-180, angle_max=180,
                        num_workers=num_workers, pin_memory=True, shuffle=True, verbose=True):
    """
    
    
    if angle is not note, applies a random rotation
    """

 
    dataloaders = {}
    
    for folder in args.folders:

        data_transform = get_transforms(args, im_mean=im_mean, im_std=im_std, angle_min=angle_min, angle_max=angle_max)

        path = os.path.join(args.root, folder) # data path
        image_dataset = torchvision.datasets.ImageFolder(path, transform=data_transform) # load the data

        dataloaders[folder] = torch.utils.data.DataLoader(
                                image_dataset, 
                                batch_size=args.batch_size if folder=='train' else args.batch_size_val,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
                        )
        if verbose: 
            print(f"Loaded {len(image_dataset)} images under {folder}")  

    return dataloaders



def enlarge_function(array, new_resolution):
    # Get the dimensions of the input array
    rows, cols = array.shape

    # Calculate the step size for each dimension
    step_x = cols / new_resolution[1]
    step_y = rows / new_resolution[0]

    # Initialize the output array with zeros
    enlarged_array = np.zeros(new_resolution)

    # Iterate over the new resolution and perform bilinear interpolation
    for i in range(new_resolution[0]):
        for j in range(new_resolution[1]):
            # Calculate the coordinates in the original array
            x = j * step_x
            y = i * step_y

            # Find the four surrounding points
            x0 = int(x)
            x1 = min(x0 + 1, cols - 1)
            y0 = int(y)
            y1 = min(y0 + 1, rows - 1)

            # Calculate the weights for bilinear interpolation
            dx = x - x0
            dy = y - y0

            # Perform bilinear interpolation
            interpolated_value = (1 - dx) * (1 - dy) * array[y0, x0] + \
                                 dx * (1 - dy) * array[y0, x1] + \
                                 (1 - dx) * dy * array[y1, x0] + \
                                 dx * dy * array[y1, x1]

            # Assign the interpolated value to the corresponding location in the enlarged array
            enlarged_array[i, j] = interpolated_value

    return enlarged_array

def to_tuple(str_size):
    return (int(str_size.split(' ')[0].split('(')[1].split(',')[0]),
            int(str_size.split(' ')[1].split(')')[0]))

def get_boxes_imagenet(box_anot):
    
    box = []
    for i in np.arange(len(box_anot)//5) :
        box.append({'ymin' : int(box_anot[(i*5)+1]),
                     'xmin' : int(box_anot[(i*5)+2]),
                     'ymax' : int(box_anot[(i*5)+3]),
                     'xmax' : int(box_anot[(i*5)+4])})
    return box

def get_mask_from_bb(target_size, orig_size, boxes):
    
    mask = np.zeros((orig_size),dtype=np.uint8) # initialize mask
    for box in boxes : 
        mask[box['ymin']:box['ymax'],box['xmin']:box['xmax']] = 1 # fill with white pixels
    return enlarge_function(mask, target_size).T


def normalize_array(arr):
    # Convert the array to float type (if not already)
    arr = arr.astype(float)
    
    # Calculate the minimum and maximum values in the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
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

    
    X_mid = math.ceil(np.mean(ground_true_indices[0]%resolution[0]))
    Y_mid = math.ceil(np.mean(ground_true_indices[0]//resolution[0]))
    
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

    return [(Y_mid , X_mid), (Y_bord , X_bord), end]

def get_like_point(heatmap, resolution, three_points):
    mid_point = heatmap[(three_points[0][0]*resolution[0]) + three_points[0][1]]
    in_point = heatmap[(three_points[1][0]*resolution[0]) + three_points[1][1]]
    ext_point = heatmap[(three_points[2][0]*resolution[0]) + three_points[2][1]]
    return mid_point, in_point, ext_point



def get_IoU(heatmap, ground_true):
    IoU = []
    for tresh in np.linspace(0,.4,10):
        n_heat = len(np.where(heatmap > tresh )[0])
        n_true = len(np.where(ground_true > 0 )[0])
        n_heat_in = len(np.where(heatmap[(heatmap > tresh) & (ground_true > 0 )])[0])
        IoU.append(n_heat_in/(n_true+ n_heat))
    return IoU

from sklearn.metrics import accuracy_score, precision_score, f1_score

def read_IoU(results_Iou):
    mean_Iou = {}
    for tresh in np.linspace(0,9,10, dtype=int):
        mean_Iou[tresh] = 0
        
    for im_ in results_Iou:
        for tresh, Iou in enumerate(im_.split(' ')) :
            mean_Iou[tresh] += float(Iou.replace(',', '').replace('[', '').replace(']', ''))
    
    for tresh in np.linspace(0,9,10, dtype=int):
        mean_Iou[tresh] /= len(results_Iou)
    print(mean_Iou)
    return mean_Iou
    
def get_best_Iou(result_Iou, best_loc):
    best_Iou = []
    for im_ in results['Iou']:
        Iou = im_.split(' ')[1]
        best_Iou.append(float(Iou.replace(',', '').replace('[', '').replace(']', '')))
    return best_Iou

def get_dist(results_dist):
    
    distances = {0:[], 1:[]}
    for dist in results_dist:
        for pos, dist_ in enumerate(dist.split(' ')) :
            distances[pos].append((float(dist_.replace(',', '').replace('[', '').replace(']', ''))))
    return [np.zeros(len(distances[0])), np.array(distances[0]), np.array(distances[1])]
    
    

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