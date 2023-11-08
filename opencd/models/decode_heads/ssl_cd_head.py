import torch.nn as nn
import torch
from .my_decode_head import MyBaseDecodeHead
from opencd.registry import MODELS
import functools
from torch import Tensor
from mmseg.utils import SampleList
from mmseg.models.utils import resize
import numpy as np
import cv2
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

bn_mom = 0.0003
class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)

class cat(torch.nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample = False):
        super(cat,self).__init__() ##parent's init func
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )
        self.conv2d=torch.nn.Sequential(
            torch.nn.Conv2d(in_chn_high + in_chn_low, out_chn, kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )
    
    def forward(self,x,y):
        # import ipdb
        # ipdb.set_trace()
        if self.do_upsample:
            x = self.upsample(x)

        x = torch.cat((x,y),1)#x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)
    
class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                torch.nn.BatchNorm2d(dim_in//2, momentum=bn_mom),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y
    
@MODELS.register_module()  
class SSL_CD_Head(MyBaseDecodeHead):
    def __init__(self, channel_list = [256, 128, 64, 32], pool_list = [True, True, True, True], **kwargs):
        super(SSL_CD_Head, self).__init__(**kwargs)

        self.decoder3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])
        
        self.df1 = DF_Module(channel_list[3], channel_list[3], True)
        self.df2 = DF_Module(channel_list[2], channel_list[2], True)
        self.df3 = DF_Module(channel_list[1], channel_list[1], True)
        self.df4 = DF_Module(channel_list[0], channel_list[0], True)

        self.catc3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.catc2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.catc1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])

        self.upsample_x2=nn.Sequential(
                        nn.Conv2d(channel_list[3],8,kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(8, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(8,1,kernel_size=3,stride=1,padding=1)
        self.conv_out_class = torch.nn.Conv2d(channel_list[3],1, kernel_size=1,stride=1,padding=0)



    def forward(self, x):
        y, ori, e = x
        print(y.shape)
        c = self.df4(x[0], x[1])


        # aff = self.affinity_map(x[0], x[1], c)

        x[0] = self.decoder3(x[0], x[2][2])
        x[1] = self.decoder3(x[1], x[3][2])
        
        c = self.catc3(c, self.df3(x[0], x[1]))


        x[0] = self.decoder2(x[0], x[2][1])
        x[1] = self.decoder2(x[1], x[3][1])
        c = self.catc2(c, self.df2(x[0], x[1]))


        x[0] = self.decoder1(x[0], x[2][0])
        x[1] = self.decoder1(x[1], x[3][0])
        c = self.catc1(c, self.df1(x[0], x[1]))


        x[0] = self.conv_out_class(x[0])
        x[1] = self.conv_out_class(x[1])
        y = self.conv_out(self.upsample_x2(c))
        

        return y
    
    def affinity_map(self, t1: Tensor, t2:Tensor, v:Tensor):
        B, C, H, W = t1.shape
        t1 = self.conv(t1)
        t2 = self.conv(t2)
        t1 = t1.view(B, C, -1).permute(0, 2, 1) # (B, 1024, 256)
        t2 = t2.view(B, C, -1).permute(0, 2, 1) # (B, 1024, 256)
        v = v.view(B, C, -1) # (B, 256, 1024)

        affinity1 = (C**-.5) * (t1 @ t1.permute(0, 2, 1))
        affinity1 = torch.sigmoid(affinity1) # (B, 1024, 1024)
        aff_map1 = (v @ affinity1).view(B, C, H, W)

        affinity2 = (C**-.5) * (t2 @ t2.permute(0, 2, 1))
        affinity2 = torch.sigmoid(affinity2)
        aff_map2 = (v @ affinity2).view(B, C, H, W)

        aff_map = aff_map2 - aff_map1
        aff_map = self.conv(aff_map)

        return aff_map
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        seg_logits = self.forward(inputs)

        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
    
    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        
        seg_logits, seg1, seg2, z, ori = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)
        
        loss = dict()

        label_size = seg_label.shape[2:]
        
        label_sized2x = (label_size[0]//2, label_size[1]//2)

        seg_logits = resize(
            input=seg_logits,
            size=label_size,
            mode='bilinear',
            align_corners=self.align_corners)
        
        max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        seg_label_down2x = max_pooling_layer(seg_label.float())

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        seg_logits = [seg_logits, seg1, seg2]
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    [seg_label.float(),seg_label_down2x.float()],
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        seg_label = seg_label.squeeze(1)
        loss['acc_seg'] = accuracy(
            seg_logits[0], seg_label, ignore_index=self.ignore_index)
        return loss

