import os 
import numpy as np 

# Torch Import
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset 

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbours

from typing import List, Tuple 


class ResNetEncd(nn.Module):
    def __init__(self, base="resnet18", proj_dim=128):
        super().__init__()

        backbone = getattr(torchvision.models, base)(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLu(inplace=True),
            nn.Linear(feat_dim, proj_dim)
        )

    def forward(self, x, return_projection: bool = True):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        if return_projection:
            proj = self.projection(feat)
            proj = F.normalize(proj, dim =1)
            return feat, proj
        return feat 
    
    def represet(self, x):
        return self.forward(x, return_projection=False)



def run_pipelin():
    encoder = None 

    os.makedirs("results", exist_ok=True)
    np.save("results/......") # adjust 
    print(f"Saved ::: Name ") # adjust 
    return None 


if __name__ == '__main__':
    pass # Run pipeline()