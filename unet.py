# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:41:48 2021

@author: Nils
"""

import torch
import torch.nn as nn


# Adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class C(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, forward=True):
        super(C, self).__init__()

        if forward:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class C2(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, forward=True):
        super(C2, self).__init__()

        #print(type(out_channels))
        #print(type(mid_channels))

        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            C(in_channels, mid_channels,forward=forward, kernel_size=3, padding=1, stride=1),
            C(mid_channels, out_channels,forward=forward, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        #print(x.size())
        #print(self.conv(x).size())
        return self.conv(x)


class C3(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels_first=None, mid_channels_second=None, forward=True):
        super(C3, self).__init__()

        if mid_channels_first is None:
            mid_channels_first = out_channels
        if mid_channels_second is None:
            mid_channels_second = out_channels

        
        self.conv = nn.Sequential(
            C(in_channels, mid_channels_first, forward=forward,kernel_size=3, padding=1, stride=(2 if forward else 1)),
            C(mid_channels_first, mid_channels_second, forward=forward,kernel_size=3, padding=1, stride=1),
            C(mid_channels_second, out_channels,forward=forward, kernel_size=3, padding=1, stride=1)
        )


    def forward(self, x):
        #print('-'*70)
        #print(self.conv(x).size())
        return self.conv(x)


def fuse( feature_maps_backward,feature_maps_forward, output_size=None):
    if output_size is not None:
        upsampled = nn.functional.interpolate(feature_maps_backward, size=output_size, mode='bilinear')
    else:
        upsampled = nn.functional.interpolate(feature_maps_backward, scale_factor=2, mode='bilinear')
    
    #print("feature_maps_backward", feature_maps_backward.size())
    #print("upsampled shape", upsampled.size())
    #print("feature_maps_forward shape", feature_maps_forward.size())
    
    fused = torch.cat((upsampled, feature_maps_forward), dim=1)  # Not entirely sure which dimension should be used.
    
    #print("fused size",fused.size())

    return fused


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        
        #print(out_channels)
        super(UNet, self).__init__()
        
        
        
        self.down64_299 = C2(in_channels, 64)
                
        self.down128_150 = C3(64, 128)
        self.down256_75 = C3(128, 256)
        self.down256_38 = C3(256, 256)
        self.down256_19 = C3(256, 256)
        #print("down128", self.down128)
        #print("down256_1", self.down256_1)
        #print("down256_2", self.down256_2)
        #print("down256_3", self.down256_3)

        self.up256_1 = C3(512, 256, forward=False)
        self.up256_2 = C3(512, 256,forward=False)
        self.up128 = C3(384, 128,forward=False)
        self.up64 = C2(192, 64,forward=False)
        self.up_final = nn.Conv2d(64, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        down64_299 = self.down64_299(x)
        down128_150 = self.down128_150(down64_299)
        down256_75 = self.down256_75(down128_150)
        down256_38 = self.down256_38(down256_75)
        down256_19 = self.down256_19(down256_38)

        fused1 = fuse(down256_19, down256_38,output_size=(38,38))
        up1 = self.up256_1(fused1)
        
        fused2 = fuse(up1, down256_75, output_size=(75,75))
        up2 = self.up256_2(fused2)
        
        fused3 = fuse(up2, down128_150, output_size=(150,150))
        up3 = self.up128(fused3)
        
        fused4 = fuse(up3, down64_299, output_size=(299,299))
        up4 = self.up64(fused4)
        
        output = self.up_final(up4)
        return output + x

