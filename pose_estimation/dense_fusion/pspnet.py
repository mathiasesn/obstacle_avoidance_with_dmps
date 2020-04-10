"""Pyramid Scene Parsing Network
"""


import torch
from torch import nn
from torch.nn import functional as F
import extractors as extractors


class PSPModule(nn.Module):
    """Pyramid Pooling Module
    """
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        """Initalization function for Pyramid Pooling Module
        
        Arguments:
            features {[type]} -- feature map from CNN
        
        Keyword Arguments:
            out_features {int} -- number of output features (default: {1024})
            sizes {tuple} -- sizes of pyramid pooling modules (default: {(1, 2, 3, 6)})
        """
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features*(len(sizes)+1), out_features, kernel_size=1)
        self.relu = nn.ReLU

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
    

    def forward(self, features):
        h, w = features.size(2), features.size(3)
        priors = [F.upsample(input=stage(features),size=(h,w),mode='bilinear') for stage in self.stages] + [features]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    """Pyramid Scene Parsing Upsample
    """
    def __init__(self, in_channels, out_channels):
        """Pyramid Scene Parsing Upsample.
        
        Arguments:
            nn {[type]} -- [description]
            in_channels {[type]} -- [description]
            out_channels {[type]} -- [description]
        """
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        """Forward propagation of PSPUpsample.
        
        Arguments:
            x {[type]} -- input
        
        Returns:
            [type] -- propagated input
        """
        return self.conv(x)


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network (PSPNet) a neural network for pixel-level
    prediction
    """
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048,
                 deep_features_size=1024, backend='resnet18', pretrained=False):
        """Pyramid Scene Parsing Network (PSPNet) a neural network for pixel-level
        prediction.
        
        Keyword Arguments:
            n_classes {int} -- number of classes (default: {21})
            sizes {tuple} -- Pyramid Pooling Module sizes (default: {(1, 2, 3, 6)})
            psp_size {int} -- (default: {2048})
            deep_features_size {int} -- (default: {1024})
            backend {str} -- Feature extractor (CNN) (default: {'resnet18'})
            pretrained {bool} -- (default: {False})
        """
        super(PSPNet, self).__init__()
        self.features = getattr(extractors, backend)()
        self.psp = PSPModule(psp_size, 1014, sizes)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.up1 = PSPUpsample(1024, 256)
        self.up2 = PSPUpsample(256, 64)
        self.up3 = PSPUpsample(64, 64)
        self.drop2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        """[summary]
        
        Arguments:
            x {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        f, class_f = self.features(x)
        p = self.psp(f)

        p = self.drop1(p)
        p = self.up1(p)

        p = self.drop2(p)
        p = self.up2(p)

        p = self.drop2(p)
        p = self.up3(p)

        return self.final(p)
