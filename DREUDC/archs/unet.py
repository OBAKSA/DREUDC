import torch
import torch.nn as nn
from einops import rearrange
import functools
import numbers


def double_conv_unet(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
    )


def convt(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    )


def conv_block_same(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 4, 3, padding=1),
    )


class UNet_raw2raw(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.PixelUnshuffle(2)
        self.upsample = nn.PixelShuffle(2)

        self.dconv_enc1_d = double_conv_unet(4, 32)
        self.dconv_enc2_d = double_conv_unet(32, 64)
        self.dconv_enc3_d = double_conv_unet(64, 128)
        self.dconv_enc4_d = double_conv_unet(128, 256)

        self.dconv_enc5_d = double_conv_unet(256, 512)

        self.dconv_enc1 = double_conv_unet(4, 32)
        self.dconv_enc2 = double_conv_unet(32, 64)
        self.dconv_enc3 = double_conv_unet(64, 128)
        self.dconv_enc4 = double_conv_unet(128, 256)

        self.maxpool = nn.MaxPool2d(2)

        self.convt1 = convt(512, 256)
        self.convt2 = convt(256, 128)
        self.convt3 = convt(128, 64)
        self.convt4 = convt(64, 32)

        self.dconv_dec1 = double_conv_unet(512, 256)
        self.dconv_dec2 = double_conv_unet(256, 128)
        self.dconv_dec3 = double_conv_unet(128, 64)
        self.dconv_dec4 = double_conv_unet(64, 32)

        self.conv_block_last = conv_block_same(32)

    def forward(self, input):
        input = self.downsample(input)
        ## Encoder ##

        # Encoder(Up)
        conv1_up = self.dconv_enc1(input)
        up1 = self.maxpool(conv1_up)

        conv2_up = self.dconv_enc2(up1)
        up2 = self.maxpool(conv2_up)

        conv3_up = self.dconv_enc3(up2)
        up3 = self.maxpool(conv3_up)

        conv4_up = self.dconv_enc4(up3)

        # Encoder(Down)
        conv1_down = self.dconv_enc1_d(input)
        down1 = self.maxpool(conv1_down)

        conv2_down = self.dconv_enc2_d(down1)
        down2 = self.maxpool(conv2_down)

        conv3_down = self.dconv_enc3_d(down2)
        down3 = self.maxpool(conv3_down)

        conv4_down = self.dconv_enc4_d(down3)
        down4 = self.maxpool(conv4_down)

        bridge = self.dconv_enc5_d(down4)

        ## Decoder ##

        # 512->256
        t1 = self.convt1(bridge)
        dec1 = torch.cat([t1, conv4_up], dim=1)
        dec1 = self.dconv_dec1(dec1)

        # 256->128
        t2 = self.convt2(dec1)
        dec2 = torch.cat([t2, conv3_up], dim=1)
        dec2 = self.dconv_dec2(dec2)

        # 128->64
        t3 = self.convt3(dec2)
        dec3 = torch.cat([t3, conv2_up], dim=1)
        dec3 = self.dconv_dec3(dec3)

        # 64->32
        t4 = self.convt4(dec3)
        dec4 = torch.cat([t4, conv1_up], dim=1)
        dec4 = self.dconv_dec4(dec4)

        # 32->3
        out = self.conv_block_last(dec4)
        out = self.upsample(out)

        return out


class UNet_raw2rgb(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.PixelUnshuffle(2)
        self.upsample = nn.PixelShuffle(2)

        self.dconv_enc1_d = double_conv_unet(4, 32)
        self.dconv_enc2_d = double_conv_unet(32, 64)
        self.dconv_enc3_d = double_conv_unet(64, 128)
        self.dconv_enc4_d = double_conv_unet(128, 256)

        self.dconv_enc5_d = double_conv_unet(256, 512)

        self.dconv_enc1 = double_conv_unet(4, 32)
        self.dconv_enc2 = double_conv_unet(32, 64)
        self.dconv_enc3 = double_conv_unet(64, 128)
        self.dconv_enc4 = double_conv_unet(128, 256)

        self.maxpool = nn.MaxPool2d(2)

        self.convt1 = convt(512, 256)
        self.convt2 = convt(256, 128)
        self.convt3 = convt(128, 64)
        self.convt4 = convt(64, 32)

        self.dconv_dec1 = double_conv_unet(512, 256)
        self.dconv_dec2 = double_conv_unet(256, 128)
        self.dconv_dec3 = double_conv_unet(128, 64)
        self.dconv_dec4 = double_conv_unet(64, 32)

        self.conv_block_last = nn.Conv2d(32, 12, 3, padding=1)

    def forward(self, input):
        input = self.downsample(input)
        ## Encoder ##

        # Encoder(Up)
        conv1_up = self.dconv_enc1(input)
        up1 = self.maxpool(conv1_up)

        conv2_up = self.dconv_enc2(up1)
        up2 = self.maxpool(conv2_up)

        conv3_up = self.dconv_enc3(up2)
        up3 = self.maxpool(conv3_up)

        conv4_up = self.dconv_enc4(up3)

        # Encoder(Down)
        conv1_down = self.dconv_enc1_d(input)
        down1 = self.maxpool(conv1_down)

        conv2_down = self.dconv_enc2_d(down1)
        down2 = self.maxpool(conv2_down)

        conv3_down = self.dconv_enc3_d(down2)
        down3 = self.maxpool(conv3_down)

        conv4_down = self.dconv_enc4_d(down3)
        down4 = self.maxpool(conv4_down)

        bridge = self.dconv_enc5_d(down4)

        ## Decoder ##

        # 512->256
        t1 = self.convt1(bridge)
        dec1 = torch.cat([t1, conv4_up], dim=1)
        dec1 = self.dconv_dec1(dec1)

        # 256->128
        t2 = self.convt2(dec1)
        dec2 = torch.cat([t2, conv3_up], dim=1)
        dec2 = self.dconv_dec2(dec2)

        # 128->64
        t3 = self.convt3(dec2)
        dec3 = torch.cat([t3, conv2_up], dim=1)
        dec3 = self.dconv_dec3(dec3)

        # 64->32
        t4 = self.convt4(dec3)
        dec4 = torch.cat([t4, conv1_up], dim=1)
        dec4 = self.dconv_dec4(dec4)

        # 32->3
        out = self.conv_block_last(dec4)
        out = self.upsample(out)

        return out


class UNet_rgb2rgb(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_enc1_d = double_conv_unet(3, 32)
        self.dconv_enc2_d = double_conv_unet(32, 64)
        self.dconv_enc3_d = double_conv_unet(64, 128)
        self.dconv_enc4_d = double_conv_unet(128, 256)

        self.dconv_enc5_d = double_conv_unet(256, 512)

        self.dconv_enc1 = double_conv_unet(3, 32)
        self.dconv_enc2 = double_conv_unet(32, 64)
        self.dconv_enc3 = double_conv_unet(64, 128)
        self.dconv_enc4 = double_conv_unet(128, 256)

        self.maxpool = nn.MaxPool2d(2)

        self.convt1 = convt(512, 256)
        self.convt2 = convt(256, 128)
        self.convt3 = convt(128, 64)
        self.convt4 = convt(64, 32)

        self.dconv_dec1 = double_conv_unet(512, 256)
        self.dconv_dec2 = double_conv_unet(256, 128)
        self.dconv_dec3 = double_conv_unet(128, 64)
        self.dconv_dec4 = double_conv_unet(64, 32)

        self.conv_block_last = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, input):
        ## Encoder ##

        # Encoder(Up)
        conv1_up = self.dconv_enc1(input)
        up1 = self.maxpool(conv1_up)

        conv2_up = self.dconv_enc2(up1)
        up2 = self.maxpool(conv2_up)

        conv3_up = self.dconv_enc3(up2)
        up3 = self.maxpool(conv3_up)

        conv4_up = self.dconv_enc4(up3)

        # Encoder(Down)
        conv1_down = self.dconv_enc1_d(input)
        down1 = self.maxpool(conv1_down)

        conv2_down = self.dconv_enc2_d(down1)
        down2 = self.maxpool(conv2_down)

        conv3_down = self.dconv_enc3_d(down2)
        down3 = self.maxpool(conv3_down)

        conv4_down = self.dconv_enc4_d(down3)
        down4 = self.maxpool(conv4_down)

        bridge = self.dconv_enc5_d(down4)

        ## Decoder ##

        # 512->256
        t1 = self.convt1(bridge)
        dec1 = torch.cat([t1, conv4_up], dim=1)
        dec1 = self.dconv_dec1(dec1)

        # 256->128
        t2 = self.convt2(dec1)
        dec2 = torch.cat([t2, conv3_up], dim=1)
        dec2 = self.dconv_dec2(dec2)

        # 128->64
        t3 = self.convt3(dec2)
        dec3 = torch.cat([t3, conv2_up], dim=1)
        dec3 = self.dconv_dec3(dec3)

        # 64->32
        t4 = self.convt4(dec3)
        dec4 = torch.cat([t4, conv1_up], dim=1)
        dec4 = self.dconv_dec4(dec4)

        # 32->3
        out = self.conv_block_last(dec4)

        return out
