"""Dataset
"""


import os
import sys
import random
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
NUM_CLASSES = len(CLASSES) + 1


class SegDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, use_noise=False):
        if mode == 'train':
            self.images = open(f'{root_dir}/train.txt', 'rt').read().split('\n')[:-1]
        else:
            self.images = open(f'{root_dir}/test.txt', 'rt').read().split('\n')[:-1]
        

        self.transform = transform

        self.image_root_dir = f'{root_dir}/rgb'
        self.mask_root_dir = f'{root_dir}/mask'

        self.counts = self.compute_class_probability()

        self.use_noise = use_noise
        self.trancolor = transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        name = self.images[index]
        img_path = os.path.join(self.image_root_dir, name + '.png')
        mask_path = os.path.join(self.mask_root_dir, name + '.png')

        img = self.load_image(path=img_path)
        gt_mask = self.load_mask(path=mask_path)

        # if self.use_noise:
        #     choice = random.randint(0, 3)
        #     if choice == 0:
        #         img = np.fliplr(img)
        #         gt_mask = np.fliplr(gt_mask)
        #     elif choice == 1:
        #         img = np.flipud(img)
        #         gt_mask = np.flipud(gt_mask)
        #     elif choice == 2:
        #         img = np.fliplr(img)
        #         img = np.flipud(img)
        #         gt_mask = np.fliplr(gt_mask)
        #         gt_mask = np.flipud(gt_mask)

        data = {
            'image': torch.from_numpy(img.astype(np.float32)),
            'mask': torch.from_numpy(gt_mask.astype(np.int64))
        }

        return data

    def compute_class_probability(self):
        counts = dict((i,0) for i in range(NUM_CLASSES))
        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + '.png')
            raw_img = Image.open(mask_path)
            imx_t = np.array(raw_img)
            imx_t[imx_t==255] = len(CLASSES)
        for i in range(NUM_CLASSES):
            counts[i] += np.sum(imx_t == i)
        return counts
    
    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)
        return torch.Tensor(p_values)
    
    def load_image(self, path):
        raw_img = Image.open(path)

        if not self.use_noise:
            raw_img = np.array(raw_img)
        else:
            raw_img = np.array(self.trancolor(raw_img))

        raw_img = np.transpose(raw_img, (2, 1, 0))
        imx_t = np.array(raw_img, dtype=np.float32) / 255.0
        return imx_t

    def load_mask(self, path):
        raw_img = Image.open(path).convert('L')
        imx_t = np.array(raw_img)
        imx_t[imx_t==255] = len(CLASSES)
        return imx_t


if __name__ == '__main__':
    data_root = 'pose_estimation/dataset/linemod/Linemod_preprocessed/data/01'

    dataset = SegDataset(root_dir=data_root, mode='train', use_noise=True)

    print(f'Class probability --> {dataset.get_class_probability()}')

    sample = dataset[0]
    img = sample['image']
    mask = sample['mask']

    img.transpose_(0, 2)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
