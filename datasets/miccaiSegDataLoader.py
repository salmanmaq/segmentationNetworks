'''
Class for loading the miccaiSeg dataset
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os

class miccaiSegDataset(Dataset):
    '''
        miccaiSeg Dataset
    '''

    def __init__(self, root_dir, transform=None):
        '''
        Args:
            root_dir (string): Directory with all the images
            transform(callable, optional): Optional transform to be applied on a sample
        '''

        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'groundtruth')
        self.image_list = [f for f in os.listdir(self.img_dir) if (f.endswith('.png') or f.endswith('.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        gt_file_name = self.image_list[idx][0:-4] + '_gt.jpg'
        gt_name = os.path.join(self.gt_dir, gt_file_name)
        image = Image.open(img_name)
        image = image.convert('RGB')
        gt = Image.open(gt_name)
        gt = gt.convert('RGB')

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)

        return image, gt
