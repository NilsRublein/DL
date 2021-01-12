import torch
import torch.nn as nn


# Adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class C(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class C2(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(C2, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            C(in_channels, mid_channels, kernel_size=3, padding=0, stride=1),
            C(mid_channels, out_channels, kernel_size=3, padding=0, stride=1)
        )

    def forward(self, x):
        return self.conv(x)


class C3(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels_first=None, mid_channels_second=None, forward=True):
        super(C3, self).__init__()

        if not mid_channels_first:
            mid_channels_first = out_channels
        if not mid_channels_second:
            mid_channels_second = out_channels

        self.conv = nn.Sequential(
            C(in_channels, mid_channels_first, kernel_size=3, padding=0, stride=(2 if forward else 1)),
            C(mid_channels_first, mid_channels_second, kernel_size=3, padding=0, stride=1),
            C(mid_channels_second, out_channels, kernel_size=3, padding=0, stride=1)
        )

    def forward(self, x):
        return self.conv(x)


def fuse(feature_maps_forward, feature_maps_backward, output_size=None):
    if not output_size:
        upsampled = nn.functional.interpolate(feature_maps_backward, size=output_size, mode='bilinear')
    else:
        upsampled = nn.functional.interpolate(feature_maps_backward, scale_factor=2, mode='bilinear')
    fused = torch.cat((upsampled, feature_maps_forward), dim=2)  # Not entirely sure which dimension should be used.
    return fused


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.down64 = C2(in_channels, 64)
        self.down128 = C3(64, 128)
        self.down256_1 = C3(128, 256)
        self.down256_2 = C3(256, 256)
        self.down256_3 = C3(256, 256)

        self.up256_1 = C3(256, 256)
        self.up256_2 = C3(256, 256)
        self.up128 = C3(256, 128)
        self.up64 = C2(128, 64)
        self.up_final = nn.Conv2d(64, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        down64 = self.down64(x)
        down128 = self.down128(down64)
        down256_1 = self.down256_1(down128)
        down256_2 = self.down256_2(down256_1)
        down256_3 = self.down256_3(down256_2)

        fused1 = fuse(down256_3, down256_2)
        up1 = self.up256_1(fused1)
        fused2 = fuse(up1, down256_1)
        up2 = self.up256_2(fused2)
        fused3 = fuse(up2, down128)
        up3 = self.up128(fused3)
        fused4 = fuse(up3, down64)
        up4 = self.up64(fused4)
        output = self.up_final(up4)
        return output


