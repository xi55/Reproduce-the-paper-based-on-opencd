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


    
@MODELS.register_module()  
class SSL_Head(MyBaseDecodeHead):
    def __init__(self, channel_list = [256, 128, 64, 32], pool_list = [True, True, True, True], **kwargs):
        super(SSL_Head, self).__init__(**kwargs)

        self.decoder3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])

        self.upsample_x2=nn.Sequential(
                        nn.Conv2d(channel_list[3],8,kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(8, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(8,1,kernel_size=3,stride=1,padding=1)
        # self.conv_out_class = torch.nn.Conv2d(channel_list[3],1, kernel_size=1,stride=1,padding=0)



    def forward(self, x):
        
        y, ori, e = x
        y = self.decoder3(y, e[2])
        y = self.decoder2(y, e[1])
        y = self.decoder1(y, e[0])
        y = self.cls_seg(y)
        # y = self.conv_out_class(y)
        # print(y.shape)
        # y = self.conv_out(self.upsample_x2(c))
        return y
    
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        seg_logits = self.forward(inputs)

        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        label_semantic_segs = [
            data_sample.label_seg_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(label_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples)
        
        # label = seg_label[0, :, :, :].permute(1, 2, 0).cpu().numpy()
        # label = label[0, :, :, :]
        # print(label.shape)
        # print(np.unique(label))
        # label = np.where(label == 1, 255, label)
        # label=np.array(label, dtype=np.uint8)
        # cv2.imshow("2", label)
        # cv2.waitKey()

        loss = dict()

        label_size = seg_label.shape[2:]

        seg_logits = resize(
            input=seg_logits,
            size=label_size,
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode



        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label.long(),
                    weight=seg_weight)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

