import numpy as np
import os

import torch
from torchvision import transforms as tt
from torch.utils.data import Dataset
from PIL import Image


class DataFolder(Dataset):
    def __init__(self, path, dataset_mean=[0.5]*3, dataset_std=[0.5]*3, ext='jpg'):
        super().__init__()
        self.path = path
        self.ext = ext
        self.mean = dataset_mean
        self.std = dataset_std

        self.files = list(filter(lambda f: f.endswith(self.ext), os.listdir(self.path)))
        self.len_ = len(self.files)

        self.transform = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(self.mean, self.std) 
        ])

        
    def __len__(self):
        return self.len_

    
    def __getitem__(self, index):
        file = os.path.join(self.path, self.files[index])
        img = Image.open(file)
        
        img = self.transform(img)
        
        return img
    
    
class VarianceSchedule:
    def __init__(self, Tmax=1000):
        self.Tmax = Tmax
        self.variance = torch.linspace(1e-4, 2e-2, self.Tmax)
        self.alpha = 1-self.variance
        self.alpha_prod = torch.cumprod(self.alpha, dim=0)
        self.alpha_prod_sqrt = torch.sqrt(self.alpha_prod)
        self.alpha_prod_inv_sqrt = torch.sqrt(1-self.alpha_prod)
        

class DiffusionDataset(DataFolder):
    def __init__(self, path, Tmax=1000, channels=3, dataset_mean=[0.5]*3, dataset_std=[0.5]*3, ext='png'):
        super().__init__(path, dataset_mean=dataset_mean, dataset_std=dataset_std, ext=ext)
    
        self.channels = channels
        self.Tmax = Tmax
        self.variance_schedule = VarianceSchedule(self.Tmax)

        
    def __getitem__(self, index):
        file = os.path.join(self.path, self.files[index])
        img = Image.open(file)
        img = self.transform(img)
        
        t = torch.randint(self.Tmax, [1]).item()
        noise = torch.normal(torch.zeros(img.shape), torch.ones(img.shape))
        alpha_prod_sqrt = self.variance_schedule.alpha_prod_sqrt[t]
        alpha_prod_inv_sqrt = self.variance_schedule.alpha_prod_inv_sqrt[t]
        noised_img = alpha_prod_sqrt*img + alpha_prod_inv_sqrt*noise
        
        return noised_img, t, noise