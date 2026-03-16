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

    train_subset = CIFAR10Subset(train_base, selected_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader 


def build_resnet18():
    model = torchvision.models.resnet18(weights=None)
    model.fc == nn.Linear(model.fc.in_features,10)
    return model 


def train_supervised(
        selected_indices,
        epochs=100, # can be reduced due to training time concerns  
        batch_size=128,
        lr=0.1, # Learning rate 
        momentum=0.9,
        weight_decay=5e-4,
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_CIFAR10Loaders(selected_indices, batch_size=batch_size)

    model = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss() # Loss function for resnet18
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max=epochs)

    

