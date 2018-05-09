'''
Class for loading the miccaiSeg dataset, the associated groundtruth images,
and determing the tool presence annotations from the groundtruth..
This is used for joint training of classification and segmentation for the
miccaiSeg dataset.
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import json
from utils import generateToolPresenceVector

class miccaiSegPlusClassDataset(Dataset):
    '''
        miccaiPlusClassSeg Dataset
    '''

    def __init__(self, root_dir, transform=None, json_path=None):
        '''
        Args:
            root_dir (string): Directory with all the training images
            transform(callable, optional): Optional transform to be applied on a sample
        '''

        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'groundtruth')
        self.image_list = [f for f in os.listdir(self.img_dir) if (f.endswith('.png') or f.endswith('.jpg'))]
        self.transform = transform

        if json_path:
            # Read the json file containing classes information
            # This is later used to generate masks from the segmented images
            self.classes = json.load(open(json_path))['classes']

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        gt_file_name = self.image_list[idx][0:-4] + '_gt.png'
        gt_name = os.path.join(self.gt_dir, gt_file_name)
        image = Image.open(img_name)
        image = image.convert('RGB')
        gt = Image.open(gt_name)
        gt = gt.convert('RGB')

        # Get the tool presense vector from the groundtruth image
        tool_presence_vector = generateToolPresenceVector(gt)

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)

        return image, gt, tool_presence_vector
