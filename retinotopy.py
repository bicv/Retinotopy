#############################################################
# data_set_type = 'focus' # Select your root between : 'boxes', 'square', 'focus', 'full', 'square'
data_set_types = ['full', 'bbox', 'focus', ]
data_set_linestyles = [':', '-.', '-', ]
import os
HOST = os.uname()[1]
# print(f'{HOST=}')
def touch(fname): open(fname, 'w').close()
# import requests
import time
tic = time.time()
from time import strftime, gmtime
datetag = strftime("%Y-%m-%d", gmtime())
datetag = '2024-05-24'
#############################################################

#############################################################
# MATPLOTLIB imports and parameters
import numpy as np
import json
from tqdm import tqdm
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
try: # before https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#removals
    cmap = plt.cm.get_cmap('viridis')
except: # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#removals
    cmap = plt.cm.colormaps['viridis']
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
else:
    device = torch.device('cpu')

# set seed function
def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
#   random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')    
#############################################################

#############################################################
data_cache = 'cached_data'
interpolation = T.InterpolationMode.BILINEAR
batch_size = 50
if '.cluster' in HOST: # mesocentre
    DATAROOT = '/scratch/lperrinet/science/Deep_learning/data'
    num_workers = 8
elif HOST in ['babbage']: # 
    DATAROOT = '/data/Deep_learning/data'
    num_workers = 2
elif HOST in ['CONEC-LID-001']: # emmy
    DATAROOT = '/data/JNJER/Deep_learning/data'
    # TODO test 
    DATAROOT = '/scratch/ImageNet'
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
    DATAROOT = '/Volumes/SSD1TO/ImageNet'
    DATAROOT = '/Volumes/data/2024_archives/2024_science/Deep_learning/data'
    interpolation = T.InterpolationMode.NEAREST
    num_workers = 4
elif 'Ahsoka' in HOST: 
    DATAROOT = '/Volumes/backups/2023_archives/2023_science/JNJER_PhD/data'
    DATAROOT = '/Volumes/data/2024_archives/2024_science/Deep_learning/data'
    num_workers = 24
    device = torch.device('cpu')
elif 'DESKTOP-27VNO0E' in HOST: 
    DATAROOT = '/mnt/d/Data/'
    num_workers = 16
else:
    raise ValueError(f'Unknown host {HOST}')

pprint(f'On date {datetag}, Running learning on host {HOST} with device {device}, pytorch=={torch.__version__}')
#############################################################

#############################################################
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass, asdict, field

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
    num_epochs: int = 2 # 
    n_train_stop: int = 0 # set to zero to use all images
    seed: int = 1998 # Set the seed for reproducibility 
    batch_size: int = batch_size # Set number of images per input batch
    batch_size_val: int = batch_size # Set number of images per input batch
    lr: float = 3.e-5 # Set learning rate 
    momentum: float = .04 # Set the momentum
    beta2: float = 0 # Set the second momentum - use SGD if set to 0
    rs_min: float = 0.00
    rs_max: float = -5.00
    do_polar: bool = True # use a retinotopic mapping
    do_scratch: bool = False # whether we use pretrained weights or not during transfer learning
    do_rotation: bool = False # just use this for rotation attacks
    
    set_seed(seed=seed, seed_torch=True)
    
args = Params()

# keep a human readable copy of the parameters
json_fname = os.path.join(data_cache, datetag + '_config_args.json')
with open(json_fname, 'wt') as f:
    json.dump(vars(args), f, indent=4)
#############################################################


#############################################################
# Wordnet
from nltk.corpus import wordnet as wn
# import nltk
# nltk.download('wordnet')
# annotations = json.load(open(args.annotations) ) 
with open(args.loader) as json_file:
    Imagenet_urls_ILSVRC_2016 = json.load(json_file)
    
# match = []
labels = []
revlabels_dico = {}
labels_dico = {}
i_labels_dico = {}
#----------------Get the label for the Imagenet categorization------------------------
for i_img, img_id in enumerate(Imagenet_urls_ILSVRC_2016):
    syn_= wn.synset_from_pos_and_offset('n', int(img_id.replace('n','')))
    sem_ = syn_.hypernym_paths()[0]
    label = syn_.name().split(".")[0]
    labels.append(label)
    i_labels_dico[label] = i_img
    labels_dico[label] = img_id
    revlabels_dico[img_id] = label
    # for i in np.arange(len(sem_)):
    # for i in np.arange(len(sem_)):
    #     if sem_[i].lemmas()[0].name() in 'animal' :
    #         match.append(i_img)
#############################################################


#############################################################
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225])

def imgs_to_np(img_list, im_mean=im_mean, im_std=im_std):
    images = torchvision.utils.make_grid(img_list)
    """Imshow for Tensor."""
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)

def imshow(img_list, im_mean=im_mean, im_std=im_std, 
           title=None, fig_height=5, fig=None, ax=None): #allow to display the input image
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_height*len(img_list), fig_height))
    inp = imgs_to_np(img_list, im_mean=im_mean, im_std=im_std)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None: fig.suptitle(title)
    fig.set_facecolor(color='white')
    plt.tight_layout()
    

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



class to_log_polar_tens:
    def __init__(self, grid):
        self.grid = grid

    def __call__(self, images):
        # images = images.to(device)
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
    

# Resnet datasets initialisation
def get_transforms(args, im_mean=im_mean, im_std=im_std, angle_min=-180, angle_max=180):

    grid = get_grid(args)#.to(device)
    mask = make_mask(args.image_size)#.to(device)

    transforms = [                
        T.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        T.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    ]   

    if args.do_rotation: # used for augmentation and testing rotations
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

def charge_model(model_name='resnet50', model_path=None, do_scratch=False, do_circular=False, verbose=True):
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
        if verbose: print(f'loading .... {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    if do_circular:
        model = make_padding_circular_again(model)

    return model
#############################################################


#############################################################
size_ratio = 0.3 # how much of the image to use relative to radius
def get_positions(image, resolution, size_ratio=size_ratio, method='full'):
    _, H, W = image.shape

    min_size = np.min((H, W))
    box_size = int(min_size*size_ratio)
    if method=='valid':
        if H < W:
            shift = (0, (W-H)/2)
        else:
            shift = ((H-W)/2, 0)

        pos_h = np.linspace(shift[0]+box_size/2, min_size+shift[0]-box_size/2, resolution[0], endpoint=True)
        pos_w = np.linspace(shift[1]+box_size/2, min_size+shift[1]-box_size/2, resolution[1], endpoint=True)
    else:
        pos_h = np.linspace(0, H, resolution[0]+2, endpoint=True)[1:-1]
        pos_w = np.linspace(0, W, resolution[1]+2, endpoint=True)[1:-1]

    pos_H, pos_W = np.meshgrid(pos_h, pos_w)

    return pos_H, pos_W, box_size

def compute_likelihood_map(args, model, image, resolution=(11, 11), # how many fixation points to use
                           size_ratio=size_ratio, # how much of the image to use relative to radius
                           N_batch=100, method='full'):

    pos_H, pos_W, box_size = get_positions(image, resolution, size_ratio, method=method)
    data_transform = get_transforms(args)
    # image = image.to(device)
    # model = model.to(device)

    N_fixations = resolution[0] * resolution[1]
    proba_label = np.zeros((N_fixations, 1000))
    for idx_start in np.arange(0, N_fixations, N_batch):
        idx_stop = np.min((idx_start+N_batch, N_fixations))
            
        with torch.no_grad():
            cropped_images = torch.empty((idx_stop-idx_start, 3, box_size, box_size), device=device)

            for i_fixation, (h, w) in enumerate(zip(pos_H.ravel()[idx_start:idx_stop], 
                                                    pos_W.ravel()[idx_start:idx_stop])):
                h, w = int(h), int(w)
                cropped_image = crop(image, h-box_size//2, w-box_size//2, box_size, box_size)
                cropped_images[i_fixation, ...] = data_transform(cropped_image)

            outputs = torch.nn.functional.softmax(model(cropped_images), dim=1)

        proba_label[idx_start:idx_stop, :] = outputs.detach().cpu().numpy()
        
    return pos_H, pos_W, proba_label
#############################################################

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
