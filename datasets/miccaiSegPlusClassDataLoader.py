'''
Class for loading the miccaiSeg dataset and the associated tool frame
classifications.
This is only used for evluation for tool classification/presense detection
for the MICCAI Tool Detection Challenge 2016 dataset.
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import json

class miccaiSegPlusClassDataset(Dataset):
    '''
        miccaiPlusClassSeg Dataset
    '''

    def __init__(self, root_dir, transform=None, json_path=None):
        '''
        Args:
            root_dir (string): Directory with all the test images
            transform(callable, optional): Optional transform to be applied on a sample
        '''

        self.root_dir = root_dir
        os.listdir(self.root_dir)
        self.sub_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(root_dir, d))
        self.image_list = []
        for sub_dir in self.sub_dirs:
            sub_list = os.listdir(os.path.join(root_dir, sub_dir))
            for f in sub_list:
                self.image_list.append(os.path.join(root_dir, sub_dir, f))
        self.transform = transform

        if json_path:
            # Read the json file containing classes information
            # This is later used to generate masks from the segmented images
            self.classes = json.load(open(json_path))['classes']

        # Read the corresponding tool presense/classification annotations
        ann_files = [f for f in os.listdir(self.root_dir) if f.endswith('.txt')]
        self.ann_list = {x: np.loadtxt(os.path.join(self.root_dir, x)) for x in ann_files}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')

        ann_file_name = img_name.split('/')[-2] + '.txt'
        frm_number = int(img_name.split('/')[-1])
        corresponding_entry = self.ann_list[ann_file_name] == frm_number
        presense_vector = self.ann_list[ann_file_name][corresponding_entry]
        presense_vector = presense_vector[1:]

        if self.transform:
            image = self.transform(image)

        return image, presense_vector
