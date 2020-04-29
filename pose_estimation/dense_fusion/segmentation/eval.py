"""Evaluation of SegNet
"""


from __future__ import print_function
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import time
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pose_estimation.dense_fusion.segmentation.dataset import SegDataset, NUM_CLASSES
from pose_estimation.dense_fusion.segmentation.segnet import Segnet
from pose_estimation.dense_fusion.segmentation.loss import Loss
from pose_estimation.dense_fusion.segmentation.utils import AverageMeter, intersection_and_union, accuracy


def visualize(img, label, pred):
    img = np.transpose(img[0], (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    label = label[0].astype(np.float32)
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    gt = cv2.addWeighted(img, 1.0, label, 0.5, 0)

    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(img, 1.0, pred, 0.5, 0)
    
    cv2.imshow('Input', img)
    cv2.imshow('Ground Truth', gt)
    cv2.imshow('Prediction', result)

    return cv2.waitKey(20)


def main(args):
    output_result_dir = f'pose_estimation/dense_fusion/segmentation/logs/{args.item}'

    dataset = SegDataset(root_dir=args.data_root, item=args.item, mode='eval')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1) # set num_workers=8
    print(f'Validation set loaded with a length of {len(dataset)}')

    model = Segnet(input_nbr=3, label_nbr=NUM_CLASSES).cuda()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    fw = open(f'{output_result_dir}/eval_result_logs.txt', 'w')

    criterion = Loss()

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    bar = tqdm(dataloader)
    for i, data in enumerate(bar, 0):
        img, label = data
        img = Variable(img).cuda()
        # mask = Variable(mask).cuda()

        t1 = time.time()
        pred = model(img)
        time_meter.update(time.time() - t1)

        pred = pred.data.cpu().detach().numpy()
        pred = pred[0][1]
        pred[pred > 0] = 1
        pred[pred < 0] = 0

        label = label.data.cpu().detach().numpy()

        acc, pix = accuracy(pred, label)
        acc_meter.update(acc, pix)

        intersection, union = intersection_and_union(pred, label, NUM_CLASSES)
        intersection_meter.update(intersection)
        union_meter.update(union)

        avg_acc = acc_meter.average() * 100
        iou = np.mean(intersection_meter.sum / union_meter.sum) * 100

        bar.set_description(f'Average acc {avg_acc:.2f} Mean IoU {iou:.2f}')
        fw.write(f'No. {i} Accuracy {acc} IoU class 0 {intersection[0]/union[0]} class 1 {intersection[1]/union[1]}\n')

        if args.show:
            img = img.data.cpu().numpy()
            key = visualize(img, label, pred)

            if key == 27:
                bar.write('Terminating because of key press')
                return

    iou = intersection_meter.sum / union_meter.sum
    for i, _iou in enumerate(iou):
        print(f'class {i} IoU {_iou:4f}')
        fw.write(f'class {i} IoU {_iou}\n')
    
    print(f'Mean IoU: {iou.mean()*100:.2f} Accuracy {acc_meter.average()*100:.2f} Average time {time_meter.average():.4f}')
    fw.write(f'Mean IoU: {iou.mean()*100} Accuracy {acc_meter.average()*100} Average time {time_meter.average()}\n')
    fw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval a SegNet model')
    parser.add_argument('--data_root', type=str, default='pose_estimation/dataset/linemod/Linemod_preprocessed', help='dataset root directory')
    parser.add_argument('--model', type=str, default='pose_estimation/dense_fusion/segmentation/trained_models/01/model_36_0.0028926266822963953.pth', help='full/path/to/trained/model')
    parser.add_argument('--item', type=str, default='01', help='item number (default: 01 for ape)')
    parser.add_argument('--show', action='store_const', const=True, default=False, help='visualise results')
    args = parser.parse_args()

    main(args)
