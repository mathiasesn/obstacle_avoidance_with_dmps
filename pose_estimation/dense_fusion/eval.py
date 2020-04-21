"""Evaluation
"""

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import random
import numpy as np
import yaml
import copy
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
from progressbar import *
from pose_estimation.dataset.linemod.dataset import PoseDataset
from pose_estimation.dense_fusion.lib.network import PoseNet, PoseRefineNet
from pose_estimation.dense_fusion.lib.loss import Loss
from pose_estimation.dense_fusion.lib.loss_refiner import Loss_refine
from pose_estimation.dense_fusion.lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from pose_estimation.dense_fusion.lib.utils import KNearestNeighbor


def main(opt):
    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = 500
    iteration = 4
    bs = 1
    dataset_config_dir = 'pose_estimation/dataset/linemod/dataset_config'
    # output_result_dir = 'pose_estimation/dense_fusion/eval_result/linemod'
    output_result_dir = 'pose_estimation/dense_fusion/logs/linemod'
    knn = KNearestNeighbor(1)

    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    testdataset = PoseDataset('eval', num_points, False, opt.dataset_root, 0.0, True)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    diameter = []
    meta_file = open(f'{dataset_config_dir}/models_info.yml', 'r')
    meta = yaml.load(meta_file, Loader=yaml.FullLoader)
    for obj in objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    print(diameter)

    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]

    fw = open(f'{output_result_dir}/eval_result_logs.txt', 'w')

    widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
    progress_bar = ProgressBar(widgets=widgets, maxval=testdataset.length).start()

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, idx = data

        if len(points.size()) == 2:
            # print(f'No.{i} NOT Pass! Lost detection!')
            widgets[0] = FormatLabel(f'No.{i} NOT Pass! Lost detection!')
            fw.write(f'No.{i} NOT Pass! Lost detection!\n')
            continue
        
        points = Variable(points).cuda()
        choose = Variable(choose).cuda()
        img = Variable(img).cuda()
        target = Variable(target).cuda()
        model_points = Variable(model_points).cuda()
        idx = Variable(idx).cuda()

        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final # quaternion
            my_t = my_t_final # translation

        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()

        if idx[0].item() in sym_list:
            pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
            target = torch.index_select(target, 1, inds.view(-1) - 1)
            dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))

        if dis < diameter[idx[0].item()]:
            success_count[idx[0].item()] += 1
            # print(f'No.{i} of {testdataset.length} Pass! Distance: {dis}')
            widgets[0] = FormatLabel(f'No.{i} Pass! Distance: {dis}')
            fw.write(f'No.{i} Pass! Distance: {dis}\n')
        else:
            # print(f'No.{i} of {testdataset.length} NOT Pass! Distance: {dis}')
            widgets[0] = FormatLabel(f'No.{i} NOT Pass! Distance: {dis}')
            fw.write(f'No.{i} NOT Pass! Distance: {dis}\n')
        
        num_count[idx[0].item()] += 1

        progress_bar.update(i+1)

    progress_bar.finish()
    
    for i in range(num_objects):
        print(f'Object {objlist[i]} success rate: {float(success_count[i]) / num_count[i]}')
        fw.write(f'Object {objlist[i]} success rate: {float(success_count[i]) / num_count[i]}\n')

    print(f'ALL success rate: {float(sum(success_count)) / sum(num_count)}')
    fw.write(f'ALL success rate: {float(sum(success_count)) / sum(num_count)}\n')

    fw.close()


if __name__ == '__main__':
    print(f'\nStarting {sys.argv[0]} with arguments:\n {sys.argv[1:]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='pose_estimation/dataset/linemod/Linemod_preprocessed', help='dataset root directory')
    parser.add_argument('--model', type=str, default='', help='PoseNet model (full path)')
    parser.add_argument('--refine_model', type=str, default='', help='PoseRefineNet model (full path)')
    opt = parser.parse_args()

    main(opt)

    print(f'\nFinished {sys.argv[0]}')