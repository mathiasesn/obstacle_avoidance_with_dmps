"""Train of DenseFusion
"""


import os
import sys
sys.path.insert(0, os.getcwd()) # sets the root to the current working directory
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
from pose_estimation.dataset.linemod.dataset import PoseDataset as PoseDataset_linemod
from pose_estimation.dense_fusion.lib.network import PoseNet, PoseRefineNet
from pose_estimation.dense_fusion.lib.loss import Loss
from pose_estimation.dense_fusion.lib.loss_refiner import Loss_refine
from pose_estimation.dense_fusion.lib.utils import setup_logger


def main(args):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    if opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'pose_estimation/dense_fusion/trained_models/linemod'
        opt.log_dir = 'pose_estimation/dense_fusion/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print(f'Unknown dataset, found: {args.dataset}')
        return

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load(f'{opt.outf}/{opt.resume_posenet}'))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load(f'{opt.outf}/{opt.resume_refinenet}'))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    if opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(f'--------- Dataset loaded! ---------')
    print(f'length of the training set: {len(dataset)}')
    print(f'length of the testing set: {len(testdataloader)}')
    print(f'number of sample points on mesh: {opt.num_points_mesh}')
    print(f'symmetry object list: {opt.sym_list}')

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))

    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger(f'epoch{epoch}', os.path.join(opt.log_dir, f'epoch_{epoch}_log.txt'))
        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        logger.info(f'Train time {time_str}, Training started')

        train_count = 0
        train_dis_avg = 0.0

        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, model_points, idx = data

                points = Variable(points).cuda()
                choose = Variable(choose).cuda()
                img = Variable(img).cuda()
                target = Variable(target).cuda()
                model_points = Variable(model_points).cuda()
                idx = Variable(idx).cuda()

                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
                    logger.info(f'Train time {time_str} Epoch {epoch} Batch {int(train_count/opt.batch_size)} Frame {train_count} Avg_dis:{train_dis_avg/opt.batch_size}')
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), f'{opt.outf}/pose_refine_model_current.pth')
                    else:
                        torch.save(estimator.state_dict(), f'{opt.outf}/pose_model_current.pth')
        
        print(f'--------- Epoch {epoch} train finished ---------')

        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        logger = setup_logger(f'epoch{epoch}_test', os.path.join(opt.log_dir, f'epoch_{epoch}_test_log.txt'))
        logger.info(f'Train time {time_str}, Training started')
        
        test_dis = 0.0
        test_count = 0

        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data

            points = Variable(points).cuda()
            choose = Variable(choose).cuda()
            img = Variable(img).cuda()
            target = Variable(target).cuda()
            model_points = Variable(model_points).cuda()
            idx = Variable(idx).cuda()

            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
            
            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()

            time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
            logger.info(f'Test time {0} Test Frame No.{test_count} dis:{dis}')

            test_count += 1
        
        test_dis = test_dis / test_count

        time_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time))
        logger.info(f'Test time {time_str} Epoch {epoch} TEST FINISH Avg dis: {test_dis}')
        
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), f'{opt.outf}/pose_refine_model_{epoch}_{test_dis}.pth')
            else:
                torch.save(estimator.state_dict(), f'{opt.outf}/pose_model_{epoch}_{test_dis}.pth')

            print(f'--------- Epoch {epoch} --> best test model saved with Avg dis: {test_dis} ---------')
        
        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        
        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

            if opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print(f'--------- Dataset loaded! ---------')
            print(f'length of the training set: {len(dataset)}')
            print(f'length of the testing set: {len(test_dataset)}')
            print(f'number of sample points on mesh: {opt.num_points_mesh}')
            print(f'symmetry object list: {opt.sym_list}')

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)


if __name__ == '__main__':
    print(f'\nStarting {sys.argv[0]} with arguments:\n {sys.argv[1:]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='linemod', help='the dataset used')
    parser.add_argument('--dataset_root', type=str, default='pose_estimation/dataset/linemod/Linemod_preprocessed', help='dataset root dir')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--w', default=0.015, help='learning rate')
    parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
    parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
    parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
    parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
    parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
    parser.add_argument('--resume_posenet', type=str, default='',  help='resume PoseNet model')
    parser.add_argument('--resume_refinenet', type=str, default='',  help='resume PoseRefineNet model')
    parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
    opt = parser.parse_args()

    main(opt)

    print(f'\nFinished {sys.argv[0]}')
