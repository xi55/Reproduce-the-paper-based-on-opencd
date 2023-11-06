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
class Fccdn_Head(MyBaseDecodeHead):
    def __init__(self, channel_list = [256, 128, 64, 32], pool_list = [True, True, True, True], **kwargs):
        super(Fccdn_Head, self).__init__(**kwargs)

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

        self.G = Generator(num_classes=2)
        self.D = PixelDiscriminator(6)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=3, stride=1, padding=1),
        #     torch.nn.BatchNorm2d(channel_list[0]),
        # )
        # self.cam = CAM(256 ** 2, 2)

    def forward(self, x):
        z1, z2 = x[0], x[1]
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
        

        return [y, x[0], x[1], [z1, z2], x[4]]
    
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


    def cgan(self, z: Tensor, ori:Tensor, seg_label: Tensor):

        x1, x2 = ori
        z1, z2 = z
        z1 = self.conv(z1)
        z2 = self.conv(z2)
        c = self.df4(z1, z2)
        aff = self.affinity_map(z1, z2, c)

        # z = torch.cat(tensors=(z1, z2), dim=1)
        fake_A = self.G(aff, x1)
        fake_B = self.G(aff, x2)
        fake_AB = torch.cat((fake_A, fake_B), 1)
        pred_fake = self.D(fake_AB.detach())

        real_AB = torch.cat((x1, x2), 1)
        pred_real = self.D(real_AB)

        pred_diff = pred_fake - pred_real

        # print(pred_diff)
        # print(torch.unique(pred_diff))

        return [pred_diff, pred_fake, pred_real, fake_B]

class CAM(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(CAM, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x        
    
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            # norm_layer(ndf * 2),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
    
class Generator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        net = []

        channels_in = [256*4, 256, 128, 64, 32]
        channels_out = [256, 128, 64, 32, 3]
        active = ["R", "R", "R", "R", "Sigmoid"]
        stride = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]
        kernel = [1, 4, 4, 4, 4]
        for i in range(len(channels_in)):
            net.append(nn.ConvTranspose2d(in_channels=channels_in[i], out_channels=channels_out[i],
                                          kernel_size=kernel[i], stride=stride[i], padding=padding[i], bias=False))
            if active[i] == "R":
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.ReLU())
            elif active[i] == "Sigmoid":
                net.append(nn.Sigmoid())

        self.generator = nn.Sequential(*net)
        # self.label_embedding = nn.Embedding(1024, 2)

    def forward(self, x, label):
        
        B, C, H, W = x.shape

        label = label.view(B, C*3, H, W)
        
        data = torch.cat(tensors=(x, label), dim=1)
        
        out = self.generator(data)
        # print(out.shape)
        return out
    
