## Sarthak Uday Talwadkar

import torch
import torch.nn as nn

from modelDef.common import default_conv, MeanShift, ResBlocks, Upsampler

'''
    It consist of 
    MeanShift layers for input normalization/denormalization
    Initial feature extraction (head)
    Main processing body with residual blocks (body)
    Upsampling and reconstruction (tail)
'''
class EDSR(nn.Module):
    def __init__(self, config):
        super(EDSR, self).__init__()

        self.sub_mean = MeanShift(config.rgb_range, config.rgb_mean)
        self.add_mean = MeanShift(config.rgb_range, config.rgb_mean, sign = 1)

        self.head = default_conv(3, config.num_feature_maps, 3)

        self.body = nn.Sequential(
            *[ResBlocks(default_conv, config.num_feature_maps, 3, config.res_scale) 
                for _ in range(config.num_resblocks)]
              )
        
        self.body.append(default_conv(config.num_feature_maps, config.num_feature_maps, 3))

        self.tail = nn.Sequential(
            Upsampler(default_conv, config.scale, config.num_feature_maps),
            default_conv(config.num_feature_maps, 3, 3)
        )

    '''
        Preprocess the image by subtracting the mean 
        Process through residual blocks and add global residual connection
        finally Upsample and reconstruct and also add mean
    '''
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
