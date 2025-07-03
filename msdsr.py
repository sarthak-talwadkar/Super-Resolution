## Sarthak Uday Talwadkar 

import torch
import torch.nn as nn

from modelDef.common import default_conv, MeanShift, ResBlocks, Upsampler

'''
    It consist of 
    MeanShift layers for input normalization/denormalization
    Initial feature extraction (head)
    Main processing body with residual blocks (body) 
    Randomly upsamples from the list of scales and reconstruct(tail)
'''

class EDSR(nn.Module):
    def __init__(self, config):
        super(EDSR, self).__init__()

        self.num_feature_maps = config.num_feature_maps

        self.sub_mean = MeanShift(config.rgb_range, config.rgb_mean)
        self.add_mean = MeanShift(config.rgb_range, config.rgb_mean, sign=1)

        self.head = default_conv(3, config.num_feature_maps, 3)

        self.body = nn.Sequential(
            *[ResBlocks(default_conv, config.num_feature_maps, 3, config.res_scale)
              for _ in range(config.num_resblocks)]
        )

        self.body.add_module('final_conv', default_conv(config.num_feature_maps, config.num_feature_maps, 3))
        
        self.upsamplers = nn.ModuleDict({
            str(scale): nn.Sequential(
                Upsampler(default_conv, scale, config.num_feature_maps),
                default_conv(config.num_feature_maps, 3, 3)
            ) for scale in [2, 3, 4]
        })

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x  
        
        x = self.upsamplers[str(scale)](res)
        x = self.add_mean(x)
        return x