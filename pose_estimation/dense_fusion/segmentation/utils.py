"""Utils for segmentation
"""

import os
import sys
import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, n):
        self.val = val
        self.sum = val
        self.count = n
        self.avg = val
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def value(self):
        return self.val
    
    def average(self):
        return self.avg


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def iou(pred, label, num_classes):
    pred = np.asarray(pred).copy()
    label = np.asarray(label).copy()
    pred += 1
    label += 1
    pred = pred * (label > 0)
    intersection = pred * (pred == label)
    area_intersection, _ = np.histogram(intersection, bins=num_classes, range=(1, num_classes))
    area_pred, _ = np.histogram(pred, bins=num_classes, range=(1, num_classes))
    area_label, _ = np.histogram(label, bins=num_classes, range=(1, num_classes))
    area_union = area_pred + area_label - area_intersection
    return area_intersection, area_union
