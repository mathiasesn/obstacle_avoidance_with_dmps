from collections import OrderedDict
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_weights_sequential(target, source_state):
    """[summary]
    
    Arguments:
        target {[type]} -- [description]
        source_state {[type]} -- [description]
    """
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """[summary]
    
    Arguments:
        in_planes {[type]} -- [description]
        out_planes {[type]} -- [description]
    
    Keyword Arguments:
        stride {int} -- [description] (default: {1})
        dilation {int} -- [description] (default: {1})
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    """[summary]
    
    Arguments:
        nn {[type]} -- [description]
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        """[summary]
        
        Arguments:
            inplanes {[type]} -- [description]
            planes {[type]} -- [description]
        
        Keyword Arguments:
            stride {int} -- [description] (default: {1})
            downsample {[type]} -- [description] (default: {None})
            dilation {int} -- [description] (default: {1})
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        """[summary]
        
        Arguments:
            x {[type]} -- [description]
        """
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottlenect(nn.Module):
    """[summary]
    
    Arguments:
        nn {[type]} -- [description]
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        """[summary]
        
        Arguments:
            inplanes {[type]} -- [description]
            planes {[type]} -- [description]
        
        Keyword Arguments:
            stride {int} -- [description] (default: {1})
            downsample {[type]} -- [description] (default: {None})
            dilation {int} -- [description] (default: {1})
        """
        super(Bottlenect, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.relu = nn.ReLu(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """[summary]
        
        Arguments:
            x {[type]} -- [description]
        """
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """[summary]
    
    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, block, layers=(3,4,23,3)):
        """[summary]
        
        Arguments:
            block {[type]} -- [description]
        
        Keyword Arguments:
            layers {tuple} -- [description] (default: {(3,4,23,3)})
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """[summary]
        
        Arguments:
            block {[type]} -- [description]
            planes {[type]} -- [description]
            blocks {[type]} -- [description]
        
        Keyword Arguments:
            stride {int} -- [description] (default: {1})
            dilation {int} -- [description] (default: {1})
        """
        downsample = None
        if stride != 1 or self.inplanes != (planes*block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, (planes*block.expansion), kernel_size=1, stride=stride, bias=False)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)
