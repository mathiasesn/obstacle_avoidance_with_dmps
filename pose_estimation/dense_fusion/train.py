"""Train of DenseFusion
"""


import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from pose_estimation.dataset.dataset import PoseDataset
from pose_estimation.dense_fusion.network import PoseNet, PoseRefineNet
from pose_estimation.dense_fusion.loss import Loss
from pose_estimation.dense_fusion.loss_refiner import LossRefine
from pose_estimation.dense_fusion.utils import setup_logger


def main(args):
    args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if args.dataset == 'linemod':
        args.num_objects = 13
        args.num_points = 500
        args.out_file = 'pose_estimation/dense_fusion/trained_models/linemod'
        args.log_dir = 'pose_estimation/dense_fusion/logs/linemod'
        args.repeat_epoch = 20
    else:
        print(f'Unknown dataset, found: {args.dataset}')
        return

    estimator = PoseNet(num_points=args.num_points, num_obj=args.num_objects)
    estimator.cuda()

    refiner = PoseRefineNet(num_points=args.num_points, num_obj=args.num_objects)
    refiner.cuda()

    if args.resume_posenet != '':
        estimator.load_state_dict(torch.load(f'{args.out_file}/{args.resume_posenet}'))

    if args.resume_refinenet != '':
        refiner.load_state_dict(torch.load(f'{args.out_file}/{args.resume_refinenet}'))
        args.refine_start = True
        args.decay_start = True
        args.lr *= args.lr_rate
        args.w *= args.w_rate
        args.batch_size = int(args.batch_size / args.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=args.lr)
    else:
        args.refine_start = False
        args.decay_start = False
        optimizer = optim.Adam(refiner.parameters(), lr=args.lr)
    
    if args.dataset == 'linemod':
        dataset = PoseDataset('train', args.num_points, True, args.dataset_root, args.noise_trans, args.refine_start)
        test_dataset = PoseDataset('test', args.num_points, False, args.dataset_root, 0.0, args.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    args.sym_list = dataset.get_sym_list()
    args.num_pts_mesh = dataset.get_num_pts_mesh()

    print(f'--------- Dataset loaded! ---------')
    print(f'length of the training set: {len(dataset)}')
    print(f'length of the testing set: {len(test_dataset)}')
    print(f'number of sample points on mesh: {args.num_pts_mesh}')
    print(f'symmetry object list: {args.sym_list}')

    criterion = Loss(args.num_pts_mesh, args.sym_list)
    criterion_refine = LossRefine(args.num_pts_mesh, args.sym_list)

    best_test = np.Inf

    if args.start_epoch == 1:
        for log in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, log))
    
    start_time = time.time()

    for epoch in range(args.start_epoch, args.max_epoch):
        logger = setup_logger(f'epoch{epoch}', os.path.join(args.log_dir, f'epoch_{epoch}_log.txt'))
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))
        logger.info(f'Train time {time_str}, Training started')
        
        train_count = 0
        train_dis_avg = 0.0

        if args.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        
        optimizer.zero_grad()

        for rep in range(args.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                pts, choose, img, target, model_pts, idx = data

                pts = Variable(pts).cuda()
                choose = Variable(choose).cuda()
                img = Variable(img).cuda()
                target = Variable(target).cuda()
                model_pts = Variable(model_pts).cuda()
                idx = Variable(idx).cuda() 

                pred_r, pred_t, pred_c, emb = estimator(img, pts, choose, idx)
                loss, dis, new_pts, new_target = criterion(pred_r, pred_t, pred_c,
                                                           target, model_pts, ids,
                                                           pts, args.w,
                                                           args.refine_start)

                if args.refine_start:
                    for i in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_pts, emb, idx)
                        dis, new_pts, new_target = criterion_refine(pred_r, pred_t,
                                                                    new_target, model_pts,
                                                                    idx, new_pts)
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % args.batch_size == 0:
                    time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))
                    logger.info(f'Train time: {time_str} Epoch: {epoch} Batch: {int(train_count / opt.batch_size)} Frame: {train_count} Avg_dis: {train_dis_avg / opt.batch_size}')
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if args.refine_start:
                        torch.save(refiner.state_dict(), f'{args.out_file}/pose_refine_model_current.pth')
                    else:
                        torch.save(estimator.state_dict(), f'{args.out_file}/pose_model_current.pth')
        
        print(f'--------- Epoch {epoch} train finished ---------')

        logger = setup_logger(f'epoch{epoch}_test', os.path.join(args.log_dir, f'epoch_{epoch}_test_log.txt'))
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))
        logger.info(f'Test time {time_str}, Testing started')

        test_dis = 0.0
        test_count = 0

        estimator.eval()
        refiner.eval()

        for j, data in enumerate(test_dataloader, 0):
            pts, choose, img, target, model_pts, idx = data

            pts = Variable(pts).cuda()
            choose = Variable(choose).cuda()
            img = Variable(img).cuda()
            target = Variable(target).cuda()
            model_pts = Variable(model_pts).cuda()
            idx = Variable(idx).cuda()

            pred_r, pred_t, pred_c, emb = estimator(img, pts, choose, idx)
            _, dis, new_pts, new_target = criterion(pred_r, pred_t, pred_c, target, model_pts,
                                                    idx, pts, args.w, args.refine_start)
            
            if args.refine_start:
                for i in rnage(0, args.iteration):
                    pred_r, pred_t = refiner(new_pts, emb, idx)
                    dis, new_pts, new_target = criterion_refine(pred_r, pred_t, new_target,
                                                                model_pts, idx, new_pts)
                    
            test_dis += dis.item()
            
            time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))
            logger.info(f'Test time: {time_str} Test Frame No.{test_count} dis: {dis}')

            test_count += 1
        
        test_dis = test_dis / test_count

        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))
        logger.info(f'Test time: {time_str} Epoch: {epoch} Test finish Avg dis: {test_dis}')

        if test_dis <= best_test:
            best_test = test_dis
            if args.refine_start:
                torch.save(refiner.state_dict(), f'{args.out_file}/pose_refine_model_{epoch}_{test_dis}.pth')
            else:
                torch.save(estimator.state_dict(), f'{args.out_file}/pose_model_{epoch}_{test_dis}.pth')

            print(f'--------- Epoch {epoch} --> best test model saved with Avg dis: {test_dis} ---------')
        
        if best_test < args.decay_margin and not args.decay_start:
            args.decay_start = True
            args.lr *= args.lr_rate
            args.w *= args.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=args.lr)
        
        if best_test < args.refine_margin and not args.refine_start:
            args.refine_start = True
            args.batch_size = int(args.batch_size / args.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=args.lr)

            if args.dataset == 'linemod':
                dataset = PoseDataset('train', args.num_points, True, args.dataset_root, args.noise_tran, args.refine_start)
                test_dataset = PoseDataset('test', args.num_points, False, args.dataset_root, 0.0, args.refine_start)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.workers)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

            args.sym_list = dataset.get_sym_list()
            args.num_pts_mesh = dataset.get_num_pts_mesh()

            print(f'--------- Dataset loaded! ---------')
            print(f'length of the training set: {len(dataset)}')
            print(f'length of the testing set: {len(test_dataset)}')
            print(f'number of sample points on mesh: {args.num_pts_mesh}')
            print(f'symmetry object list: {args.sym_list}')

            criterion = Loss(args.num_pts_mesh, args.sym_list)
            criterion_refine = LossRefine(args.num_pts_mesh, args.sym_list)


if __name__ == '__main__':
    print(f'\nStarting {sys.argv[0]} with arguments:\n {sys.argv[1:]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='linemod', help='dataset')
    parser.add_argument('--dataset_root', type=str, default='pose_estimation/dataset/Linemod_preprocessed', help='dataset root dir')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_rate', type=float, default=0.3, help='learning rate decay rate')
    parser.add_argument('--w', type=float, default=0.015, help='learning rate')
    parser.add_argument('--w_rate', type=float, default=0.3, help='learning rate decay rate')
    parser.add_argument('--decay_margin', type=float, default=0.016, help='margin to decay lr and w')
    parser.add_argument('--refine_margin', type=float, default=0.013, help='margin to start the training of iterative refinement')
    parser.add_argument('--noise_trans', type=float, default=0.03, help='range of the random noise of translation added to the training data')
    parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
    parser.add_argument('--max_epoch', type=int, default=500, help='max number of epochs to train')
    parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
    parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
    parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
    args = parser.parse_args()

    main(args)

    print(f'\nFinished {sys.argv[0]}')
