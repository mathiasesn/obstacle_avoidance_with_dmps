"""Train SegNet
"""


from __future__ import print_function
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import time
import torch
from progressbar import *
from torch.utils.data import DataLoader
from pose_estimation.dense_fusion.segmentation.dataset import SegDataset, NUM_CLASSES
from pose_estimation.dense_fusion.segmentation.segnet import Segnet
from pose_estimation.dense_fusion.segmentation.loss import Loss


def main(args):
    data_root = args.data_root

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    dataset = SegDataset(root_dir=data_root, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    test_dataset = SegDataset(root_dir=data_root, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    if CUDA:
        model = Segnet(in_channels=3, out_channels=1).cuda()
        class_weights = 1.0 / dataset.get_class_probability().cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
        # criterion = Loss().cuda()
    else:
        model = Segnet(in_channels=3, out_channels=1)
        class_weights = 1.0 / dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # criterion = Loss()

    if args.resume_model != '':
        model.load_state_dict(torch.load(args.resume_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_cost = float('inf')

    model.train()

    for epoch in range(args.n_epochs):
        print(f'Starting training...')

        train_loss_f = 0.0
        train_time = 0
        st_time = time.time()

        f = open(f'{args.log_dir}/epoch_{epoch}_log.txt', 'w')
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        f.write(f'Train time: {time_str}, Training started\n')
        
        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
        progress_bar = ProgressBar(widgets=widgets, maxval=len(dataset)).start()

        for batch in dataloader:
            in_tensor = torch.autograd.Variable(batch['image'])
            t_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                in_tensor = in_tensor.cuda()
                t_tensor = t_tensor.cuda()

            pred_tensor, softmax_tensor = model(in_tensor)

            optimizer.zero_grad()
            loss = criterion(pred_tensor, t_tensor)
            loss.backward()
            optimizer.step()

            train_loss_f += loss.float()
            pred_f = softmax_tensor.float()

            time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
            f.write(f'Train time {time_str} Batch {train_time} CEloss {loss.float()}\n')

            if train_time != 0 and train_time % 1000 == 0:
                torch.save(model.state_dict(), f'{args.save_dir}/model_current.pth')

            train_time += 1

            widgets[0] = FormatLabel(f'Epoch {epoch} Batch {train_time} CEloss {loss.float()}')
            progress_bar.update(i+1)
        
        progress_bar.finish()
        
        train_loss_f = train_loss_f / train_time
        
        f.write(f'Train finished average CEloss: {train_loss_f}\n')
        f.close()

        print(f'Strating testing...')

        model.eval()

        test_loss_f = 0.0
        test_time = 0

        f = open(f'{args.log_dir}/epoch_{epoch}_test_log.txt', 'w')
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        f.write(f'Test time: {time_str}, Testing started\n')
                
        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
        progress_bar = ProgressBar(widgets=widgets, maxval=len(test_dataset)).start()

        for batch in dataloader:
            in_tensor = torch.autograd.Variable(batch['image'])
            t_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                in_tensor = in_tensor.cuda()
                t_tensor = t_tensor.cuda()

            pred_tensor, softmax_tensor = model(in_tensor)

            optimizer.zero_grad()
            loss = criterion(pred_tensor, t_tensor)
            loss.backward()
            optimizer.step()

            test_loss_f += loss.float()
            pred_f = softmax_tensor.float()

            time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
            f.write(f'Test time {time_str} Batch {test_time} CEloss {loss.float()}\n')

            test_time += 1

            widgets[0] = FormatLabel(f'Epoch {epoch} Batch {test_time} CEloss {loss.float()}')
            progress_bar.update(i+1)
        
        progress_bar.finish()

        test_loss_f = test_loss_f / test_time
        
        f.write(f'Test finished average CEloss {test_loss_f}\n')
        f.close()

        if test_all_cost <= best_val_cost:
            best_val_cost = test_all_cost
            torch.save(model.state_dict(), f'{args.save_dir}/model_{epoch}_{test_loss_f}.pth')
            print(f'NEW MODEL SAVED AT EPOCH {epoch} WITH AVERAGE CELOSS {test_loss_f:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a SegNet model')
    parser.add_argument('--data_root', type=str, default='pose_estimation/dataset/linemod/Linemod_preprocessed/data/01', help='dataset root dir')
    parser.add_argument('--save_dir', type=str, default='pose_estimation/dense_fusion/segmentation/trained_models/ape', help='path to save models')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 3)')
    parser.add_argument('--n_epochs', type=int, default=6000, help='epochs to train (default: 600)')
    parser.add_argument('--workers', type=int, default=1, help='nunber of data loading workers (default: 8)') # change to 8
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--log_dir', type=str, default='pose_estimation/dense_fusion/segmentation/logs/ape', help='path to save logs')
    parser.add_argument('--gpu', type=int, default='0', help='gpu id')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path')
    args = parser.parse_args()

    main(args)
