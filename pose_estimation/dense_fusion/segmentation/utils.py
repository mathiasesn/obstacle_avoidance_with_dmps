"""Utils for segmentation
"""

import os
import sys
import numpy as np


class AverageMeter():
    """Average meter is a class used to store values
    """
    def __init__(self):
        """Creates a AverageMeter object an sets values to zero
        """
        self.reset()

    def reset(self):
        """Sets all values in AverageMeter to zero
        """
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, n):
        """Intialize values in AverageMeter

        Arguments:
            val {float} -- value
            n {int} -- count
        """
        self.val = val
        self.sum = val
        self.count = n
        self.avg = val
        self.initialized = True

    def update(self, val, n=1):
        """Update AverageMeter

        Arguments:
            val {float} -- value

        Keyword Arguments:
            n {int} -- count (default: {1})
        """
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        """Add value to AverageMeter

        Arguments:
            val {float} -- value
            n {int} -- count
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def value(self):
        """Get value

        Returns:
            float -- value
        """
        return self.val
    
    def average(self):
        """Get average

        Returns:
            float -- average
        """
        return self.avg


def accuracy(pred, label):
    """Calculate the accuracy of a prediction and the corresponding label

    Arguments:
        pred {np.array} -- prediction
        label {np.array} -- label

    Returns:
        float, int -- accuracy, number of pixel
    """
    valid = (pred >= 0)
    # pred[pred > 0] = 1 
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersection_and_union(pred, label, num_classes):
    """Calculate the intersection and union

    Arguments:
        pred {np.array} -- prediction
        label {np.array} -- label
        num_classes {int} -- number of classes

    Returns:
        np.array -- intersection and union
    """
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
