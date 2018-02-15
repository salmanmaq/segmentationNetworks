'''
Class for loading the MICCAI dataset
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os

class miccaiDataset(Dataset):
    '''
        MICCAI Dataset
    '''

    def __init__(self, root_dir, transform=None):
        '''
        Args:
            root_dir (string): Directory with all the images
            transform(callable, optional): Optional transform to be applied on a sample
        '''

        self.root_dir = root_dir
        self.image_list = [f for f in os.listdir(root_dir) if (f.endswith('.png') or f.endswith('.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
