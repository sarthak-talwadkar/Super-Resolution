## Sarthak Uday Talwadkar

import torch
import torch.nn as nn
import math

'''
    Standard convolution with 'same' padding (kernel size must be odd)
    Parameters : 
    in_channels: Input channel dimension
    out_channels: Output channel dimension
    kernel_size: Convolution kernel size (must be odd for proper padding)
    bias
'''
def default_conv(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding = (kernel_size // 2),
        bias = bias
    )

'''
    Normalization layer: Subtracts/adds dataset mean using fixed 1x1 convolution
    Parameters : 
    rgb_range: Input value range (typically 255)
    rgb_mean: Dataset mean values (per channel)
    rgb_std: Dataset std values (per channel)
    sign: -1 for subtraction (preprocessing), 1 for addition (postprocessing)
'''
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std = (1.0, 1.0, 1.0), sign = -1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)

        std = torch.Tensor(rgb_std)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1)/std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean)/std

        self.requires_grad = False

'''
    EDSR-style residual block with scaling factor, no batch normalization
    Parameters :
    conv: Convolution function to use
    num_feature_maps: Number of feature channels
    kernel_size: Convolution kernel size
    res_scale: Residual scaling factor 
'''
class ResBlocks(nn.Module):
    def __init__(self, conv, num_feature_maps, kernel_size, res_scale):
        super(ResBlocks, self).__init__()

        self.res_scale = res_scale

        self.body =  nn.Sequential(
            conv(num_feature_maps, num_feature_maps, kernel_size),
            nn.ReLU(True),
            conv(num_feature_maps, num_feature_maps, kernel_size)
            )

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

'''
    Multi-scale upsampler using pixel shuffle (supports 2x, 4x, 8x etc. scales)
    Parameters : 
    conv: Convolution function to use
    scale: Upsampling factor (must be power of 2)
    num_feature_maps: Number of input feature channels
'''
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, num_feature_maps):
        m = []

        if(scale & (scale - 1)) == 0:

            num_upsamples = int(math.log(scale, 2))
            for _ in range(num_upsamples):
                m.append(conv(num_feature_maps, 4 * num_feature_maps, 3))
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU(inplace = True))
        else : 
            raise NotImplementedError("Scale {scale} not supported. Use power-of-2 scales.")

        super(Upsampler, self).__init__(*m)
