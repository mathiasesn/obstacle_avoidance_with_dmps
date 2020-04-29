"""Dataset
"""


import os
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from collections import Counter

CLASSES = (
    'background',
    'ape'
)
NUM_CLASSES = len(CLASSES)


class SegDataset(Dataset):
    """SegDataset
    """
    def __init__(self, root_dir, item, mode='train', transform=None, use_noise=False):
        """Creates a SegDataset object

        Arguments:
            root_dir {string} -- path/to/root/dir

        Keyword Arguments:
            mode {str} -- train, test or eval (default: {'train'})
            transform {[type]} -- ?? (default: {None})
            use_noise {bool} -- adds noise to images if true (default: {False})
        """
        self.mode = mode

        if mode == 'train':
            self.images = open(f'{root_dir}/data/{item}/train.txt', 'rt').read().split('\n')[:-1]
            # self.images = open(f'{root_dir}/train.txt', 'rt').read().split('\n')[:-1]
        else:
            self.images = open(f'{root_dir}/data/{item}/test.txt', 'rt').read().split('\n')[:-1]
            # self.images = open(f'{root_dir}/test.txt', 'rt').read().split('\n')[:-1]

        self.image_root_dir = f'{root_dir}/data/{item}/rgb'

        if mode == 'eval':
            self.mask_root_dir = f'{root_dir}/segnet_results/{item}_label'
        else:
            self.mask_root_dir = f'{root_dir}/data/{item}/mask'
        
        self.transform = transform
        self.use_noise = use_noise
        self.trancolor = transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05)

    def __len__(self):
        """Get length of dataset

        Returns:
            int -- length of dataset
        """
        return len(self.images)
    
    def __getitem__(self, index):
        """Get item

        Arguments:
            index {int} -- index

        Returns:
            Tensor, Tensor -- image and mask
        """
        name = self.images[index]
        img_path = os.path.join(self.image_root_dir, name + '.png')
        if self.mode == 'eval':
            mask_path = f'{self.mask_root_dir}/{name}_label.png'
        else:
            mask_path = os.path.join(self.mask_root_dir, name + '.png')

        img = self.load_image(path=img_path)
        mask = self.load_mask(path=mask_path)

        if self.use_noise:
            choice = random.randint(0, 3)
            if choice == 0:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            elif choice == 1:
                img = np.flipud(img)
                mask = np.flipud(mask)
            elif choice == 2:
                img = np.fliplr(img)
                img = np.flipud(img)
                mask = np.fliplr(mask)
                mask = np.flipud(mask)
        
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask!=0] = 1

        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64))

        return img, mask
    
    def load_image(self, path):
        """Load image

        Arguments:
            path {string} -- path/to/image

        Returns:
            np.array -- image
        """
        img = Image.open(path).convert('RGB')
        if not self.use_noise:
            img = np.array(img)
        else:
            img = np.array(self.trancolor(img))
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA) 
        # img = np.transpose(img, (2, 0, 1))
        img = np.array(img, dtype=np.float32) / 255.0
        return img

    def load_mask(self, path):
        """Load mask

        Arguments:
            path {string} -- path/to/mask

        Returns:
            np.array -- mask
        """
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        # # imx_t[imx_t==255] = len(CLASSES)
        # img[img==255] = 1
        return img


if __name__ == '__main__':
    data_root = 'pose_estimation/dataset/linemod/Linemod_preprocessed/data/01'
    dataset = SegDataset(root_dir=data_root, mode='train', use_noise=True)

    for i, data in enumerate(dataset):
        img, mask = data
        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
        mask = mask.numpy().astype(np.float32)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        dst = cv2.addWeighted(img, 1.0, mask, 0.5, 0)
        cv2.imshow('Image', img)
        cv2.imshow('Mask', mask)
        cv2.imshow('Image with mask', dst)

        key = cv2.waitKey(0)
        if key == 27:
            break
