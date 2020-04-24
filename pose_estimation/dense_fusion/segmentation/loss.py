"""Loss for SegNet
"""


import math
import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable


cross_entropy_loss = nn.CrossEntropyLoss()


def loss_calculation(semantic, target):
    """Loss calculation for SegNet

    Arguments:
        semantic {torch.Tensor} -- prediction
        target {torch.Tensor} -- ground truth

    Returns:
        float -- loss
    """
    bs = semantic.size()[0]
    pix_num = 480 * 640

    target = target.view(bs, -1).view(-1).contiguous()
    semantic = semantic.view(bs, 14, pix_num).transpose(1, 2).contiguous().view(bs * pix_num, 14).contiguous()
    semantic_loss = cross_entropy_loss(semantic, target)

    return semantic_loss


class Loss(_Loss):
    """Loss for SegNet

    Arguments:
        _Loss {loss._Loss} -- Base class for all neural network modules
    """
    def __init__(self):
        """Creates a object of Loss
        """
        super(Loss, self).__init__(True)
    
    def forward(self, semantic, target):
        """Calculates the Loss

        Arguments:
            semantic {torch.Tensor} -- prediction
            target {torch.Tensor} -- ground truth

        Returns:
            float -- loss
        """
        return loss_calculation(semantic, target)