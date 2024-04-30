import torch
import numpy as np
from skimage.transform import resize

from utils import *

# Gradient-weighted Affinity Activation Mapping
# Yang, Z., Zhong, W., Zhao, L., & Yu-Chian Chen, C. (2022). 
# MGraphDTA: deep multiscale graph neural network for explainable drug-target binding affinity prediction. 
# Chemical science, 13(3), 816–833. https://doi.org/10.1039/d1sc05180f
class GradAAM():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(self.save_hook)
        self.target_feat = None

    def save_hook(self, md, fin, fout):
        self.target_feat = fout.x

    def __call__(self, data):
        self.model.eval()
        output = self.model(data).view(-1)
        grad = torch.autograd.grad(output, self.target_feat)[0]

        channel_weight = torch.mean(grad, dim=0, keepdim=True)
        channel_weight = normalize(channel_weight)
        weighted_feat = self.target_feat * channel_weight

        cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
        cam = normalize(cam)
        return output.detach().cpu().numpy(), cam


# Gradient-weighted Target Affinity Activation Mapping (Novel contribution)
# Extension of GradCAM based approach for regression involving 1D Sequential data
class GradTAM():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(self.save_hook)
        self.target_feat = None

    def save_hook(self, md, fin, fout):
        self.target_feat = fout

    def __call__(self, data):
        self.model.eval()
        output = self.model(data).view(-1)
        grad = torch.autograd.grad(output, self.target_feat)[0]
        
        channel_weight = torch.mean(grad, dim=0, keepdim=True)
        channel_weight = normalize(channel_weight)
        weighted_feat = self.target_feat * channel_weight

        cam = torch.sum(weighted_feat[0], dim=-1).detach().cpu().numpy()
        cam = normalize(resize(cam, (1200,)))
        return output.detach().cpu().numpy(), cam