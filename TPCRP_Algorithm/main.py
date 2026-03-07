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
    
    def represent(self, x):
        return self.forward(x, return_projection=False)

class ConstructiveNTXent(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward_pass(self, z_a: torch.Tensor, z_b : torch.Tensor) -> torch.Tensro:

        batch = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)

        sim_matrix = torch.matmul(z, z.T) / self.temperature

        diag_mask = torch.eye(2 * batch ,  device=z.device).bool()

        sim_matrix.masked_fill_(diag_mask, -9e15)

        positives = torch.arrange(batch, 2 * batch, device = z.device)
        labels = positives.repeat(2) # Cross-entropy target to be satisfied 

        loss = F.cross_entropy(sim_matrix, labels)
        return loss 

def run_pipeline():
    encoder = None 

    os.makedirs("results", exist_ok=True)
    np.save("results/......") # adjust 
    print(f"Saved ::: Name ") # adjust 
    return None 


if __name__ == '__main__':
    pass # Run pipeline()