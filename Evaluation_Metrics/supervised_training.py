import time # To measure time elapsed 

import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader, Subset 



class CIFAR10Subset(Dataset):
    pass 


def get_CIFAR10Loaders(selected_indices, batch_size=128):
    pass 


def build_resnet18():
    pass 

