import time # To measure time elapsed 

import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader, Subset 



class CIFAR10Subset(Dataset):
    

    def __init__(self, base_dataset, indices):
        self.base = base_dataset 
        self.indices = indices 


    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base[real_idx]


def get_CIFAR10Loaders(selected_indices, batch_size=128):
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorrizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    transform_test = transforms.Compoes([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    train_base = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms_train
    )
    test_set = torch.vision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_test
    )


def build_resnet18():
    pass 

