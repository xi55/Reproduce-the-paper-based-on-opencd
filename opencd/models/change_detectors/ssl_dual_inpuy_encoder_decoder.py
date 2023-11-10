# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
import torch
from torch import Tensor

from opencd.registry import MODELS
from .dual_input_encoder_decoder import DIEncoderDecoder
from .siamencoder_decoder import SiamEncoderDecoder

@MODELS.register_module()
class SDIEncoderDecoder(SiamEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema = EMA(self.backbone, 0.99)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_from, img_to, img_seg = torch.split(inputs, self.backbone_inchannels, dim=1)

        # feat_from = self.backbone(img_from)
        # feat_to = self.backbone(img_to)
        feat_seg = self.backbone(img_seg)
        self.ema.update()
        with torch.no_grad():
            feat_from = self.backbone(img_from)
            feat_to = self.backbone(img_to)
        self.ema.restore()

        

        if self.with_neck:
            x = self.neck([feat_from, feat_to])
            x_seg = self.neck(feat_seg)
        x.append(x_seg)
        return x
    
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.cd_decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
            """Calculate losses from a batch of inputs and data samples.

            Args:
                inputs (Tensor): Input images.
                data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                    It usually includes information such as `metainfo` and
                    `gt_sem_seg`.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """

            x = self.extract_feat(inputs)
            # x.append(x_seg)
            # print(len(x))
            losses = dict()
            loss_decode = self._cd_decode_head_forward_train(x, data_samples)
            losses.update(loss_decode)
            
            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(x, data_samples)
                losses.update(loss_aux)

            return losses
    
class EMA:
    def __init__(self, model, decay):
        self.model = model.to('cuda:0')
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
