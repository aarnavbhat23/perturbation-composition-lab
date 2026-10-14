import torch
from torch import nn


def symmetric_features(first,second):
    return torch.cat([first+second,torch.abs(first-second),first*second],dim=-1)


class ResidualComposer(nn.Module):
    def __init__(self,latent=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(latent*3,128),nn.GELU(),nn.Dropout(.2),
                                     nn.Linear(128,64),nn.GELU(),nn.Linear(64,latent))

    def forward(self,first,second):
        return self.network(symmetric_features(first,second))
