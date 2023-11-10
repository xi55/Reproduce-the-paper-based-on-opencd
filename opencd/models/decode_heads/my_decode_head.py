# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch.nn.functional as F
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from mmseg.models.builder import build_loss
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

import numpy as np
import cv2

class MyBaseDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg. Default: None.
        threshold (float): Threshold for binary segmentation in the case of
            `num_classes==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, **kwargs):
          super(MyBaseDecodeHead, self).__init__(**kwargs)
        
       

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        # print(feat.shape)
        output = self.conv_seg(feat)
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    def add_gaussian_noise(self, tensor, mean=0, std=1):
        noise = torch.randn(tensor.size(), device=0) * std + mean
        noisy_tensor = tensor + noise
        return noisy_tensor

    def cgan(self, 
             z: Tensor, 
             ori: Tensor,
             seg_label: Tensor):
        pass

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, seg1, seg2, z, ori = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)
        # print(ori[0].shape)
        # pred_diff, pred_fake, pred_real, fake_B = self.cgan(z, ori, seg_label)

        # label = seg_label[0, :, :, :].permute(1, 2, 0).cpu().numpy()
        # # label = label[0, :, :, :]
        # print(label.shape)
        # print(np.unique(label))
        # label = np.where(label == 1, 255, label)
        # label=np.array(label, dtype=np.uint8)
        # cv2.imshow("2", label)
        # cv2.waitKey()
        
        loss = dict()

        label_size = seg_label.shape[2:]
        
        label_sized2x = (label_size[0]//2, label_size[1]//2)

        seg_logits = resize(
            input=seg_logits,
            size=label_size,
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg1 = resize(
            input=seg1,
            size=label_sized2x,
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg2 = resize(
            input=seg2,
            size=label_sized2x,
            mode='bilinear',
            align_corners=self.align_corners)
        
        # seg_label_down2x = cv2.resize(seg_label, (256, 256))
        max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        seg_label_down2x = max_pooling_layer(seg_label.float())

        # print(seg_label_down2x)
        # print(seg_label_down2x.shape)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        # seg_label = seg_label.squeeze(1)

        # print(seg_weight)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        # loss['loss_D'] = (self.D_loss(pred_fake, False) + self.D_loss(pred_real, True))/2
        # loss['loss_l1'] = self.L1_loss(fake_B, ori[1])
        # loss['loss_G'] = (self.G_loss(pred_fake, True) + loss['loss_l1'])

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

        

        # fake_l = pred_diff * loss['loss_l1']
        fake_l = torch.sigmoid(seg_logits[0])
        fake_l = fake_l[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
        # print(fake_l)
        # print(fake_l.shape)
        # print(np.unique(fake_l))
        fake_l = np.where(fake_l < 0.5, 0, fake_l)
        fake_l = np.where(fake_l >= 0.5, 255, fake_l)
        fake_l=np.array(fake_l, dtype=np.uint8)
        cv2.imwrite('D:/git/open-cd-main/logs/swincd/1/vis/' + batch_data_samples[0].seg_map_path.split('\\')[-1], fake_l)
        
        # cv2.waitKey()

        
        
        

        loss['acc_seg'] = accuracy(
            seg_logits[0], seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, seg1, seg2, z, ori = seg_logits
        if(isinstance(seg_logits, list)):
            res=[]
            for seg in [seg_logits, seg1, seg2]:
                print(type(seg))
                s = resize(
                    input=seg,
                    size=batch_img_metas[0]['img_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners)
                res.append(s)
            return res

        else:
            seg_logits = resize(
                    input=seg_logits,
                    size=batch_img_metas[0]['img_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners)
            return seg_logits
        
