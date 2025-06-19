import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.autograd import Variable
from layers import *
from data import coco, voc
import os

class MobileNetSSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # MobileNetV2 base from torchvision
        self.base = base.features
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        print("Layer 6 channels:", self.base[6].out_channels)  # Should be 32
        print("Layer 13 channels:", self.base[13].out_channels)  # Should be 96

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        print("Head configurations:")
        for i, layer in enumerate(self.loc):
            print(f"Head {i}: in={layer.in_channels}, out={layer.out_channels}")

        print("Extra layers output channels:")
        for i, layer in enumerate(self.extras):
            print(f'Layer={i}:', layer.out_channels)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        for k in range(len(self.base)):
            x = self.base[k](x)
            # print(f"Base layer {k} output shape: {x.shape}")
            if k in [6, 13, 18]:  # Capture layers 6 and 13 explicitly
                sources.append(x)

        # Forward through extras
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # Add every other extra layer
                sources.append(x)

        # for idx, src in enumerate(sources):
        #     print(f"Feature map {idx} shape: {src.shape}")

        # Apply multibox heads
        # print('multibox heads')
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            output = self.detect.apply(
                self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors.type(type(x.data))
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext in ('.pkl', '.pth'):
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, 
                             map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# In add_extras_mobilenet()
def add_extras_mobilenet(cfg, i):
    layers = []
    in_channels = i
    k = 0
    while k < len(cfg):
        if cfg[k] == 'S':
            layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=3, stride=2, padding=1)]
            in_channels = cfg[k+1]
            k += 2
        else:
            # Regular 1x1 convolution
            layers += [nn.Conv2d(in_channels, cfg[k], kernel_size=1)]
            in_channels = cfg[k]
            k += 1
    return layers

def multibox_mobilenet(base, extras, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    source_layers = [6, 13, 18]  # MobileNetV2 feature layers

    # Base network heads - use separate cfg entries for each base layer
    for idx, sl in enumerate(source_layers):
        in_channels = base[sl].out_channels  # 32 for layer6, 96 for layer13
        loc_layers += [nn.Conv2d(in_channels, cfg[idx] * 4, 3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, cfg[idx] * num_classes, 3, padding=1)]

    # Extra layers heads - start from cfg[2]
    for k, v in enumerate(extras[1::2], start=len(source_layers)):
        if k >= len(cfg):
            break
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, 3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, 3, padding=1)]

    return loc_layers, conf_layers

# extras_mobilenet = {
#     '300': [
#         256, 'S', 512,  # 19x19 -> 10x10
#         128, 256,  # 10x10 -> 5x5
#         128, 'S', 256, 'S',  # 5x5 -> 3x3
#         128, 'S', 256   # 3x3 -> 1x1
#     ],
#     'input_channels': 1280
# }

extras_mobilenet = {
    '300': [
        128, 'S', 256,      # 10x10 -> 5x5
        128, 'S', 256, 'S',      # 5x5 -> 3x3
        128, 'S', 256       # 3x3 -> 1x1
    ],
    'input_channels': 1280
}

mbox_mobilenet = {
    '300': [4, 6, 6, 6, 4, 4],  # First 2 values for base layers (32ch,96ch)
}

def build_mobilenet_ssd(phase, size=300, num_classes=21):
    if phase not in ["test", "train"]:
        raise ValueError(f"Invalid phase: {phase}")
    if size != 300:
        raise ValueError(f"Unsupported size: {size}")

    # Load pretrained MobileNetV2
    base_net = mobilenet_v2(pretrained=True)
    
    # Build extras
    extras = add_extras_mobilenet(extras_mobilenet['300'], 1280)
    
    # Build multibox heads
    loc, conf = multibox_mobilenet(base_net.features, extras, 
                                 mbox_mobilenet['300'], num_classes)

    return MobileNetSSD(phase, size, base_net, extras, (loc, conf), num_classes)
