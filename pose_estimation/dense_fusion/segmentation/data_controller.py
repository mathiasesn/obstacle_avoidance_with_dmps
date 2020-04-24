"""Data controller for SegNet
"""


import os
import sys
sys.path.insert(0, os.getcwd())
import copy
import random
import cv2
import numpy as np
import numpy.ma as ma
import scipy.misc
import scipy.io as scio
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from PIL import Image, ImageEnhance, ImageFilter


class SegDataset(data.Dataset):
    """SegNet Dataset
    """
    def __init__(self, root_dir, txt_list, use_noise, length):
        """Create a SegDataset object

        Arguments:
            root_dir {string} -- root directory
            txt_list {[type]} -- 
            use_noise {bool} -- if true use noise
            length {int} -- length
        """
        self.path = []
        self.real_path = []
        self.use_noise = use_noise
        self.root = root_dir

        f = open(txt_list)
        while True:
            line = f.readline()
            if not line:
                break
            if line[-1:] == '\n':
                line = line[:-1]
            self.path.append(copy.deepcopy(line))
            if line[:5] == 'data/':
                self.real_path.append(copy.deepcopy(line))
        f.close()

        self.length = length
        self.data_len = len(self.path)
        self.back_len = len(self.real_path)

        self.trancolor = transforms.ColorJitter(brightness=0.2,
                                                contrast=0.2,
                                                saturation=0.2,
                                                hue=0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.back_front = np.array([[1 for i in range(640)] for j in range(480)])
    
    def __getitem__(self, idx):
        """Get item

        Arguments:
            idx {int} -- index

        Returns:
            torch.Tensor, torch.Tensor -- rgb image and target
        """
        index = random.randint(0, self.data_len - 10)

        label = np.array(Image.open(f'{self.root}/{self.path[index]-label.png}'))
        meta = scio.loadmat(f'{self.root}/{self.path[index]}-meta.mat')

        img_path_idx = f'{self.root}/{self.path[index]}-color.png'
        if not self.use_noise:
            rgb = np.array(Image.open(img_path_idx).convert('RGB'))
        else:
            rgb = np.array(self.trancolor(Image.open(img_path_idx).convert('RGB')))
        
        if self.path[index][:8] == 'data_syn':
            rgb = Image.open(img_path_idx).convert('RGB')
            rgb = ImageEnhance.Brightness(rgb).enhance(1.5).filter(ImageFilter.GaussianBlur(radius=0.8))
            rgb = np.array(self.trancolor(rgb))
            
            seed = random.randint(0, self.back_len - 10)
            img_path_seed = f'{self.root}/{self.path[seed]}-color.png'
            back = np.array(self.trancolor(Image.open(img_path_seed).convert('RGB')))
            back_label = np.array(Image.open(img_path_seed))
            
            mask = ma.getmaskarray(ma.masked_equal(label, 0))
            
            back = np.transpose(back, (2, 0, 1))
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb = rgb + np.random.normal(loc=0.0, scale=5.0, size=rgb.shape)
            rgb = back * mask + rgb
            label = back_label * mask + label
            
            rgb = np.transpose(rgb, (1, 2, 0))

            cv2.imshow('rgb', rgb)
            cv2.imshow('label', label)
        
        if self.use_noise:
            choice = random.randint(0, 3)
            if choice == 0:
                rgb = np.fliplr(rgb)
                label = np.fliplr(label)
            elif choice == 1:
                rgb = np.flipud(rgb)
                label = np.flipud(label)
            elif choice == 2:
                rgb = np.fliplr(rgb)
                rgb = np.flipud(rgb)
                label = np.fliplr(label)
                label = np.flipud(label)
        
        obj = meta['cls_indexes'].flatten().astype(np.int32)
        obj = np.append(obj, [0], axis=0)
        target = copy.deepcopy(label)

        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = self.norm(torch.from_numpy(rgb.astype(np.float32)))
        target = torch.from_numpy(target.astype(np.int64))

        return rgb, target
    
    def __len__(self):
        """Get length

        Returns:
            int -- lenght of object
        """
        return self.length