"""Train SegNet
"""


from __future__ import print_function
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import time
import torch
import numpy as np
from progressbar import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pose_estimation.dense_fusion.segmentation.dataset import SegDataset, NUM_CLASSES
from pose_estimation.dense_fusion.segmentation.segnet import Segnet
from pose_estimation.dense_fusion.segmentation.loss import Loss


def main(args):
    data_root = args.data_root

    dataset = SegDataset(root_dir=data_root, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    print(f'Training set loaded! Length of training set --> {len(dataset)}')

    test_dataset = SegDataset(root_dir=data_root, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    print(f'Test set loaded! Length of test set --> {len(test_dataset)}')

    model = Segnet(input_nbr=3, label_nbr=NUM_CLASSES).cuda()

    if args.resume_model != '':
        model.load_state_dict(torch.load(args.resume_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_cost = np.Inf
    criterion = Loss()
    st_time = time.time()

    for epoch in range(args.n_epochs):
        print(f'Starting epoch {epoch}')

        f = open(f'{args.log_dir}/epoch_{epoch}_log.txt', 'w')
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        f.write(f'Train time: {time_str}, Training started\n')
        print(f'Train time: {time_str}, Training started')

        model.train()

        train_loss_f = 0.0
        train_time = 0

        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
        progress_bar = ProgressBar(widgets=widgets, maxval=len(dataset)*10).start()

        i = 0
        for rep in range(10):
            for data in dataloader:
                img, mask = data
                img = Variable(img).cuda()
                mask = Variable(mask).cuda()
                pred = model(img)
                optimizer.zero_grad()
                loss = criterion(pred, mask)
                train_loss_f += loss.float()
                loss.backward()
                optimizer.step()

                time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
                f.write(f'Train time {time_str} Batch {train_time} CEloss {loss.float()}\n')

                if train_time != 0 and train_time % 1000 == 0:
                    torch.save(model.state_dict(), f'{args.save_dir}/model_current.pth')

                train_time += 1

                widgets[0] = FormatLabel(f'Epoch {epoch} Batch {train_time} Avg CEloss {(train_loss_f/train_time):.8f}')
                progress_bar.update(i+1)
                
                i += 1
        
        progress_bar.finish()
        
        train_loss_f = train_loss_f / train_time
        
        print(f'Train finished average CEloss: {train_loss_f}')
        f.write(f'Train finished average CEloss: {train_loss_f}\n')
        f.close()

        f = open(f'{args.log_dir}/epoch_{epoch}_test_log.txt', 'w')
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        f.write(f'Test time: {time_str}, Testing started\n')
        print(f'Test time: {time_str}, Testing started')

        model.eval()

        test_loss_f = 0.0
        test_time = 0

        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
        progress_bar = ProgressBar(widgets=widgets, maxval=len(test_dataset)).start()

        for j, data in enumerate(test_dataloader):
            img, mask = data
            img = Variable(img).cuda()
            mask = Variable(mask).cuda()
            pred = model(img)
            optimizer.zero_grad()
            loss = criterion(pred, mask)
            test_loss_f += loss.float()
            loss.backward()
            optimizer.step()

            time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
            f.write(f'Test time {time_str} Batch {test_time} CEloss {loss.float()}\n')

            test_time += 1

            widgets[0] = FormatLabel(f'Epoch {epoch} Batch {test_time} Avg CEloss {(test_loss_f/test_time):.8f}')
            progress_bar.update(j+1)
        
        progress_bar.finish()

        test_loss_f = test_loss_f / test_time
        
        print(f'Test finished average CEloss {test_loss_f}')
        f.write(f'Test finished average CEloss {test_loss_f}\n')
        f.close()

        if test_loss_f <= best_val_cost:
            best_val_cost = test_loss_f
            torch.save(model.state_dict(), f'{args.save_dir}/model_{epoch}_{test_loss_f}.pth')
            print(f'New model save at epoch {epoch} with average CEloss {test_loss_f:.8f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a SegNet model')
    parser.add_argument('--data_root', type=str, default='pose_estimation/dataset/linemod/Linemod_preprocessed/data/01', help='dataset root dir')
    parser.add_argument('--save_dir', type=str, default='pose_estimation/dense_fusion/segmentation/trained_models/ape', help='path to save models')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 3)')
    parser.add_argument('--n_epochs', type=int, default=6000, help='epochs to train (default: 600)')
    parser.add_argument('--workers', type=int, default=8, help='nunber of data loading workers (default: 8)') # change to 8
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--log_dir', type=str, default='pose_estimation/dense_fusion/segmentation/logs/ape', help='path to save logs')
    parser.add_argument('--resume_model', type=str, default='pose_estimation/dense_fusion/segmentation/trained_models/ape/model_current.pth', help='resume model path')
    args = parser.parse_args()

    main(args)
