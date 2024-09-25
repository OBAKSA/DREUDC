import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import numbers
from einops import rearrange


### projection ###
class PatchProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.proj(x)

        return x


#### LAYERNORM ####
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


### Resizing modules ###
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


### FSAS ###
### We thank the authors of "Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring (CVPR 2023)" for their codes. ###
class FrequencyAttention(nn.Module):
    def __init__(self, dim, bias):
        super(FrequencyAttention, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


### FFN ###
class FeedForwardNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.proj_in = nn.Conv2d(channels, channels * 4, kernel_size=1, stride=1, padding=0)
        self.activation = nn.GELU()
        self.proj_out = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.proj_out(x)

        return x


### Restoration Block : DREBlock ###
class TransformerBlock_Freq_Embed(nn.Module):
    def __init__(self, channels, bias, LayerNorm_type):
        super().__init__()

        # self.norm1 = LayerNorm2d(channels)
        self.norm1 = LayerNorm(channels, LayerNorm_type)
        self.attention = FrequencyAttention(channels, bias)

        # self.norm2 = LayerNorm2d(channels)
        self.norm2 = LayerNorm(channels, LayerNorm_type)
        self.ffn = FeedForwardNetwork(channels)

        self.alpha = nn.Sequential(
            nn.Conv2d(256, channels, 1, 1, 0, bias=False),
        )

    def forward(self, x, rep):
        rep = rep.unsqueeze(-1).unsqueeze(-1)
        alpha = self.alpha(rep)

        x = x + alpha * self.attention(self.norm1(x))
        out = x + self.ffn(self.norm2(x))

        return out


######################################################################
### Restoration Network ###
class Former_Freq_Embed(nn.Module):
    def __init__(self, channel=32):
        super().__init__()

        bias = False
        num_blocks = [5, 5, 6]
        LayerNorm_type = 'WithBias'

        self.pack = nn.PixelUnshuffle(2)
        self.unpack = nn.PixelShuffle(2)

        self.channel4to12 = nn.Conv2d(4, 12, kernel_size=1, stride=1, padding=0)

        self.input_projection = PatchProjection(12, channel)

        ### Down ###
        self.TransformerBlock_down1 = nn.ModuleList(
            [TransformerBlock_Freq_Embed(channel, bias, LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(channel)
        self.TransformerBlock_down2 = nn.ModuleList(
            [TransformerBlock_Freq_Embed(channel * 2, bias, LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(channel * 2)
        self.TransformerBlock_down3 = nn.ModuleList(
            [TransformerBlock_Freq_Embed(channel * 4, bias, LayerNorm_type) for i in range(num_blocks[2])])

        ### Up ###
        self.TransformerBlock_up3 = nn.ModuleList(
            [TransformerBlock_Freq_Embed(channel * 4, bias, LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(channel * 4)
        self.channel_reduce2 = nn.Conv2d(channel * 4, channel * 2, kernel_size=1, bias=bias)
        self.TransformerBlock_up2 = nn.ModuleList(
            [TransformerBlock_Freq_Embed(channel * 2, bias, LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(channel * 2)
        self.channel_reduce1 = nn.Conv2d(channel * 2, channel, kernel_size=1, bias=bias)
        self.TransformerBlock_up1 = nn.ModuleList(
            [TransformerBlock_Freq_Embed(channel, bias, LayerNorm_type) for i in range(num_blocks[0])])

        self.output_projection = PatchProjection(channel, 12)

    def forward(self, input, rep):
        ## input ##
        input = self.pack(input)

        features = self.channel4to12(input)
        temp = features

        features = self.input_projection(features)
        for layer in self.TransformerBlock_down1:
            features = layer(features, rep)
        enc1 = features

        down1 = self.down1_2(enc1)
        for layer in self.TransformerBlock_down2:
            down1 = layer(down1, rep)
        enc2 = down1

        down2 = self.down2_3(enc2)
        for layer in self.TransformerBlock_down3:
            down2 = layer(down2, rep)
        enc3 = down2

        for layer in self.TransformerBlock_up3:
            enc3 = layer(enc3, rep)
        dec3 = enc3

        up2 = self.up3_2(dec3)
        y2 = torch.cat([up2, enc2], 1)
        y2 = self.channel_reduce2(y2)
        for layer in self.TransformerBlock_up2:
            y2 = layer(y2, rep)
        dec2 = y2

        up1 = self.up2_1(dec2)
        y1 = torch.cat([up1, enc1], 1)
        y1 = self.channel_reduce1(y1)
        for layer in self.TransformerBlock_up1:
            y1 = layer(y1, rep)
        dec1 = y1

        out = self.output_projection(dec1) + temp
        out = self.unpack(out)
        return out
