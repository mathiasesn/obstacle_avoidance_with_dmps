"""SegNet
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable


class Conv2dBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride,
                 padding, bias=True, dilation=1, is_batchnorm=True):
        super(Conv2dBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod,
                nn.BatchNorm2d(int(n_filters)),
                nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
    
    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class SegnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetDown2, self).__init__()
        self.conv1 = Conv2dBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2dBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
    
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetDown3, self).__init__()
        self.conv1 = Conv2dBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2dBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = Conv2dBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = Conv2dBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = Conv2dBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class SegnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = Conv2dBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = Conv2dBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = Conv2dBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class Segnet(nn.Module):
    def __init__(self, n_classes=14, in_channels=3, is_unpooling=True):
        super(Segnet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SegnetDown2(in_channels, 64)
        self.down2 = SegnetDown2(64, 128)
        self.down3 = SegnetDown3(128, 256)
        self.down4 = SegnetDown3(256, 512)
        self.down5 = SegnetDown3(512, 512)

        self.up5 = SegnetUp3(512, 512)
        self.up4 = SegnetUp3(512, 256)
        self.up3 = SegnetUp3(256, 128)
        self.up2 = SegnetUp2(128, 64)
        self.up1 = SegnetUp2(64, n_classes)

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1
    
    def init_vgg16_params(self, vgg16):
        blocks = [
            self.down1,
            self.down2,
            self.down3,
            self.down4,
            self.down5
        ]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit
                ]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit
                ]
            
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)
        
        assert len(vgg_layers) == len(merged_layers), 'layer length does not match'

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size(), 'weight size does not match'
                assert l1.bias.size() == l2.bias.size(), 'bias size does not match'
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


# class Segnet(nn.Module):
#     def __init__(self, input_nbr=3, label_nbr=14):
#         """Creates a SegNet object

#         Arguments:
#             nn {nn.Module} -- basic module implementation in pytorch
#             input_nbr {int} -- number of inputs
#             label_nbr {int} -- number of labels
#         """
#         super(Segnet, self).__init__()

#         batch_norm_momentum = 0.1

#         self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
#         self.bn11 = nn.BatchNorm2d(64, momentum=batch_norm_momentum)
#         self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12 = nn.BatchNorm2d(64, momentum=batch_norm_momentum)

#         self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn21 = nn.BatchNorm2d(128, momentum=batch_norm_momentum)
#         self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22 = nn.BatchNorm2d(128, momentum=batch_norm_momentum)

#         self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn31 = nn.BatchNorm2d(256, momentum=batch_norm_momentum)
#         self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32 = nn.BatchNorm2d(256, momentum=batch_norm_momentum)
#         self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33 = nn.BatchNorm2d(256, momentum=batch_norm_momentum)

#         self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn41 = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42 = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43 = nn.BatchNorm2d(512, momentum=batch_norm_momentum)

#         self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51 = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52 = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53 = nn.BatchNorm2d(512, momentum=batch_norm_momentum)

#         self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53d = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52d = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51d = nn.BatchNorm2d(512, momentum=batch_norm_momentum)

#         self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43d = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42d = nn.BatchNorm2d(512, momentum=batch_norm_momentum)
#         self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.bn41d = nn.BatchNorm2d(256, momentum=batch_norm_momentum)

#         self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33d = nn.BatchNorm2d(256, momentum=batch_norm_momentum)
#         self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32d = nn.BatchNorm2d(256, momentum=batch_norm_momentum)
#         self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
#         self.bn31d = nn.BatchNorm2d(128, momentum=batch_norm_momentum)

#         self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22d = nn.BatchNorm2d(128, momentum=batch_norm_momentum)
#         self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.bn21d = nn.BatchNorm2d(64, momentum=batch_norm_momentum)

#         self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12d = nn.BatchNorm2d(64, momentum=batch_norm_momentum)
#         self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
    
#     def forward(self, x):
#         """Forward of SegNet

#         Arguments:
#             x {[type]} -- input

#         Returns:
#             [type] -- [description]
#         """
#         x11 = F.relu(self.bn11(self.conv11(x)))
#         x12 = F.relu(self.bn12(self.conv12(x11)))
#         x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

#         x21 = F.relu(self.bn21(self.conv21(x1p)))
#         x22 = F.relu(self.bn22(self.conv22(x21)))
#         x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

#         x31 = F.relu(self.bn31(self.conv31(x2p)))
#         x32 = F.relu(self.bn32(self.conv32(x31)))
#         x33 = F.relu(self.bn33(self.conv33(x32)))
#         x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

#         x41 = F.relu(self.bn41(self.conv41(x3p)))
#         x42 = F.relu(self.bn42(self.conv42(x41)))
#         x43 = F.relu(self.bn43(self.conv43(x42)))
#         x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

#         x51 = F.relu(self.bn51(self.conv51(x4p)))
#         x52 = F.relu(self.bn52(self.conv52(x51)))
#         x53 = F.relu(self.bn53(self.conv53(x52)))
#         x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)

#         x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
#         x53d = F.relu(self.bn53d(self.conv53d(x5d)))
#         x52d = F.relu(self.bn52d(self.conv52d(x53d)))
#         x51d = F.relu(self.bn51d(self.conv51d(x52d)))

#         x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
#         x43d = F.relu(self.bn43d(self.conv43d(x4d)))
#         x42d = F.relu(self.bn42d(self.conv42d(x43d)))
#         x41d = F.relu(self.bn41d(self.conv41d(x42d)))

#         x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
#         x33d = F.relu(self.bn33d(self.conv33d(x3d)))
#         x32d = F.relu(self.bn32d(self.conv32d(x33d)))
#         x31d = F.relu(self.bn31d(self.conv31d(x32d)))

#         x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
#         x22d = F.relu(self.bn22d(self.conv22d(x2d)))
#         x21d = F.relu(self.bn21d(self.conv21d(x22d)))

#         x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
#         x12d = F.relu(self.bn12d(self.conv12d(x1d)))
#         x11d = self.conv11d(x12d)

#         return x11d
