
import torch
import torch.nn as nn
import math
from models.quant_layer import *


# Gradual squeeze configurations - add intermediate layers to smooth the 512->8->512 transition
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # Original sudden squeeze: 512 -> 8 -> 512
    'VGG16_quant': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 8, 512, 'M', 512, 512, 512, 'M'],
    # Gradual squeeze: 512 -> 128 -> 32 -> 8 -> 32 -> 128 -> 512
    'VGG16_quant_gradual': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 128, 32, 8, 32, 128, 512, 'M', 512, 512, 512, 'M'],
    'VGG16': ['F', 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Bottleneck channel sizes for special handling
BOTTLENECK_CHANNELS = {8, 32, 128}  # Channels in the squeeze/expand path


class VGG_quant_gradual(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_quant_gradual, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        # Track if we're in the bottleneck region (between first 512 and next 512 after squeeze)
        in_bottleneck = False
        
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'F':  # This is for the 1st layer
                layers += [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True)]
                in_channels = 64
            elif x in BOTTLENECK_CHANNELS:
                # Bottleneck layers - use ReLU without BatchNorm for the squeeze path
                # Check if we're squeezing down (current > x) or expanding (current < x)
                if in_channels > x:
                    # Squeezing down
                    in_bottleneck = True
                    layers += [QuantConv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                elif in_channels < x:
                    # Expanding back up
                    layers += [QuantConv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                    if x >= 128:  # Add BatchNorm when expanding back to larger sizes
                        # Remove the ReLU we just added and add BN+ReLU instead
                        layers.pop()  # Remove ReLU
                        layers += [nn.BatchNorm2d(x),
                                   nn.ReLU(inplace=True)]
                else:
                    # Same size - just regular conv
                    layers += [QuantConv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
            else:
                layers += [QuantConv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                in_bottleneck = False
                
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()
    

def VGG16_quant_gradual(**kwargs):
    model = VGG_quant_gradual(vgg_name='VGG16_quant_gradual', **kwargs)
    return model



