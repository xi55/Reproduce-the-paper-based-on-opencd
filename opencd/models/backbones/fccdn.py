from typing import List, Union
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, Sequential
from torch.nn import functional as F

from opencd.registry import MODELS

bn_mom = 0.0003

class double_conv(torch.nn.Module):
    def __init__(self,in_chn, out_chn, stride=1, dilation=1):#params:in_chn(input channel of double conv),out_chn(output channel of double conv)
        super(double_conv,self).__init__() ##parent's init func

        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(in_chn,out_chn, kernel_size=3, stride=stride, dilation=dilation, padding=dilation),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1,padding=1),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        x = self.conv(x)
        return x

class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        #x_se = self.avg_pool(x)
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, use_se=False, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        first_planes = planes
        outplanes = planes * self.expansion

        self.conv1 = double_conv(inplanes, first_planes)
        self.conv2 = double_conv(first_planes, outplanes, stride=stride, dilation=dilation)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.downsample = torch.nn.MaxPool2d(stride=2,kernel_size=2) if downsample else None
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        residual = out
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.ReLU(out)

        return out
    

@MODELS.register_module()   
class FCCDN(nn.Module):
    def __init__(self, 
                 num_band, 
                 os=16, 
                 use_se=False, 
                 dilation_list = [1, 1, 1, 1],
                 stride_list = [2, 2, 2, 2],
                 pool_list = [True, True, True, True],
                 channel_list = [256, 128, 64, 32],
                 **kwargs):
        super(FCCDN, self).__init__()
        # if os >= 16:
        #     dilation_list = [1, 1, 1, 1]
        #     stride_list = [2, 2, 2, 2]
        #     pool_list = [True, True, True, True]
        # elif os == 8:
        #     dilation_list = [2, 1, 1, 1]
        #     stride_list = [1, 2, 2, 2]
        #     pool_list = [False, True, True, True]
        # else:
        #     dilation_list = [2, 2, 1, 1]
        #     stride_list = [1, 1, 2, 2]
        #     pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        # channel_list = [256, 128, 64, 32]
        # encoder
        self.block1 = BasicBlock(num_band, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])


    def forward(self, x1, x2):
        
        ori = [x1, x2]
        e1_1 = self.block1(x1)
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)

        e1_2 = self.block1(x2)
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        e1 = [e1_1, e2_1, e3_1]
        e2 = [e1_2, e2_2, e3_2]

        return [y1, y2, e1, e2, ori]