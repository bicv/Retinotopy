import torch
import torch.nn.functional as nnf
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
import torch.nn as nn

import numpy as np


if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Running on metal', device)
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU : ', torch.cuda.get_device_name(), '#GPU=', torch.cuda.device_count())
else:
    device = torch.device('cpu')


class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()

        self.resnet = model
        
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)
        
        # average pooling layer
        self.avgpool = self.resnet.avgpool
        
        # classifier
        self.classifier = self.resnet.fc
        
        # gradient placeholder
        self.gradient = None
    
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forward(self, x):
        
        # extract the features
        x = self.features(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        
        return x


def get_Grad_cam(model, data, label):
        # set the evaluation mode
        resnet = ResNet(model).to(device)
        resnet = resnet.eval()
        
        # get the image
        

        pred = resnet(data)

        pred.argmax(dim=1)  # prints tensor([2])
        
        # get the gradient of the output with respect to the parameters of the model

        pred[:, label].backward()

        
        # pull the gradients out of the model
        gradients = resnet.get_gradient()
        
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # get the activations of the last convolutional layer
        activations = resnet.get_activations(data).detach()
        
        # weight the channels by corresponding gradients
        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        GradCam = torch.mean(activations, dim=1).squeeze().cpu()
        
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        GradCam = np.maximum(GradCam, 0)
        
        # normalize the heatmap
        GradCam /= torch.max(GradCam)

        

        return GradCam, pred.argmax(dim=1).item()


def display_grad(image, likelihood_map, resolution):
    likelihood_map = enlarge_function(np.array(likelihood_map), (256, 256))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    image_lin_display = cv2.resize(image, likelihood_map.T.shape, interpolation= cv2.INTER_LINEAR)
    sns.heatmap(likelihood_map, linewidth = 0 , annot = False, cmap='coolwarm', vmin=0, cbar = True, alpha=.5)
    #ax.imshow(image_lin_display)
    ax.imshow(image_lin_display)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout();