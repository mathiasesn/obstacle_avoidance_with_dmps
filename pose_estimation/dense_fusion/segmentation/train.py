"""Train SegNet
"""

import os
import sys
sys.path.insert(0, os.getcwd())
import copy
import random
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import scipy.misc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from PIL import Image
from progressbar import *
from pose_estimation.dense_fusion.segmentation.data_controller import SegDataset
from pose_estimation.dense_fusion.segmentation.loss import Loss
from pose_estimation.dense_fusion.segmentation.segnet import SegNet as segnet


def main(args):
    args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    dataset = SegDataset(args.dataset_root, f'{args.dataset_root}/train.txt', True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    test_dataset = SegDataset(args.dataset_root, f'{args.dataset_root}/train.txt', False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    print(f'Length of dataset: {len(dataset)}')
    print(f'Length of test dataset: {len(test_dataset)}')

    model = segnet()
    model = model.cuda()

    if args.resume_model != '':
        checkpoint = torch.load(f'{args.model_save_pth}/{args.resume_model}')
        model.load_state_dict(checkpoint)
    # else:
    #     checkpoint = torch.load('pose_estimation/dense_fusion/segmentation/vgg16/vgg16-397923af.pth')
    #     model.load_state_dict(checkpoint)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = Loss()
    best_val_cost = np.Inf
    st_time = time.time()

    for epoch in range(1, args.n_epochs):
        model.train()
        train_all_cost = 0.0
        train_time = 0

        f = open(f'{args.log_dir}/epoch_{epoch}_log.txt', 'w')
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        f.write(f'Train time: {time_str}, Training started\n')
        
        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
        progress_bar = ProgressBar(widgets=widgets, maxval=len(dataset)).start()

        for i, data in enumerate(dataloader, 0):
            rgb, target = data
            rgb = Variable(rgb).cuda()
            target = Variable(target).cuda()
            
            semantic = model(rgb)

            optimizer.zero_grad()

            semantic_loss = criterion(semantic, target)
            train_all_cost += semantic_loss.item()

            semantic_loss.backward()
            optimizer.step()

            time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
            f.write(f'Train time {time_str} Batch {train_time} CEloss {semantic_loss.item()}\n')

            if train_time != 0 and train_time % 1000 == 0:
                torch.save(model.state_dict(), f'{args.model_save_pth}/model_current.pth')
            
            train_time += 1

            widgets[0] = FormatLabel(f'Batch {train_time} CEloss {semantic_loss.item()}')
            progress_bar.update(i+1)
        
        progress_bar.finish()
        
        train_all_cost = train_all_cost / train_time
        
        f.write(f'Train finished average CEloss: {train_all_cost}\n')
        f.close()

        model.eval()

        test_all_cost = 0.0
        test_time = 0

        f = open(f'{args.log_dir}/epoch_{epoch}_test_log.txt', 'w')
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        f.write(f'Test time: {time_str}, Testing started\n')
                
        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
        progress_bar = ProgressBar(widgets=widgets, maxval=len(test_dataset)).start()

        for j, data in enumerate(test_dataloader, 0):
            rgb, target = data
            rgb = Variable(rgb).cuda()
            target = Variable(target).cuda()

            semantic = model(rgb)
            
            semantic_loss = criterion(semantic, target)
            test_all_cost += semantic_loss.item()

            time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
            f.write(f'Test time {time_str} Batch {test_time} CEloss {semantic_loss.item()}\n')

            test_time += 1

            widgets[0] = FormatLabel(f'Batch {test_time} CEloss {semantic_loss.item()}')
            progress_bar.update(i+1)
        
        progress_bar.finish()

        test_all_cost = test_all_cost / test_time
        
        f.write(f'Test finished average CEloss {test_all_cost}\n')
        f.close()

        if test_all_cost <= best_val_cost:
            best_val_cost = test_all_cost
            torch.save(model.state_dict(), f'{args.model_save_pth}/model_{epoch}_{test_all_cost}.pth')
            print(f'NEW MODEL SAVED AT EPOCH {epoch} WITH AVERAGE CELOSS {test_all_cost:.6f}')


if __name__ == '__main__':
    print(f'Starting {sys.argv[0]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='pose_estimation/dataset/linemod/Linemod_preprocessed/data/01', help='dataset root dir')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size (default: 3)')
    parser.add_argument('--n_epochs', type=int, default=600, help='epochs to train (default: 600)')
    parser.add_argument('--workers', type=int, default=1, help='nunber of data loading workers (default: 8)') # change to 8
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--logs_path', type=str, default='pose_estimation/dense_fusion/segmentation/logs/ape', help='path to save logs')
    parser.add_argument('--model_save_pth', type=str, default='pose_estimation/dense_fusion/segmentation/trained_models/ape', help='path to saved models')
    parser.add_argument('--log_dir', type=str, default='pose_estimation/dense_fusion/segmentation/logs/ape', help='path to save logs')
    parser.add_argument('--resume_model', type=str, default='', help='resume model name')
    args = parser.parse_args()

    main(args)

    print(f'Finished {sys.argv[0]}')