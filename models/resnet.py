import torch
import torch.nn as nn

import torchvision.models as models

from models.utils import *
from misc.losses import *


# base model ResNet: https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.cbam = CBAM( planes, 16 ) if use_cbam else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.cbam = CBAM( planes * 4, 16 ) if use_cbam else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


model_dict = {'resnet18':  {'pretrained': models.resnet18(pretrained=True),  'block': BasicBlock, 'layers': [2, 2, 2, 2]},
              'resnet34':  {'pretrained': models.resnet34(pretrained=True),  'block': BasicBlock, 'layers': [3, 4, 6, 3]},
              'resnet50':  {'pretrained': models.resnet50(pretrained=True),  'block': Bottleneck, 'layers': [3, 4, 6, 3]},
              'resnet101': {'pretrained': models.resnet101(pretrained=True), 'block': Bottleneck, 'layers': [3, 4, 23, 3]},}

class resnet(nn.Module):
    def __init__(self, name='resnet18', n_classes=28, num_cooc_classes=5, num_recon_channels=4, load_pretrained=True, use_cbam=False):
        super(resnet, self).__init__()

        self.name = name
        self.n_classes = n_classes
        self.use_cbam = use_cbam

        block = model_dict[self.name]['block']
        layers = model_dict[self.name]['layers']

        self.inplanes = 64
        conv1 = nn.Conv2d(4, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = nn.Sequential(conv1, bn1, relu, maxpool)
        self.layer1 = self._make_layer(block, 64,  layers[0], use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_cbam=use_cbam)

        # Auxiliary decoder
        self.cbr_aux4 = nn.Sequential(conv2DBatchNormRelu(in_channels=512*block.expansion, n_filters=256*block.expansion, k_size=3, padding=1, stride=1, bias=False),
                                      conv2DBatchNormRelu(in_channels=256*block.expansion, n_filters=128*block.expansion, k_size=3, padding=1, stride=1, bias=False))
        self.cbr_aux3 = nn.Sequential(conv2DBatchNormRelu(in_channels=128*block.expansion, n_filters=64*block.expansion, k_size=3, padding=1, stride=1, bias=False),
                                      conv2DBatchNormRelu(in_channels=64*block.expansion, n_filters=32*block.expansion, k_size=3, padding=1, stride=1, bias=False))
        self.cbr_aux2 = nn.Sequential(conv2DBatchNormRelu(in_channels=32*block.expansion, n_filters=16*block.expansion, k_size=3, padding=1, stride=1, bias=False),
                                      conv2DBatchNormRelu(in_channels=16*block.expansion, n_filters=8*block.expansion, k_size=3, padding=1, stride=1, bias=False))

        # Classifiers
        self.classification = nn.Linear(512*block.expansion, self.n_classes, bias=True)
        self.classification_cooc = nn.Linear(512*block.expansion, num_cooc_classes, bias=True) # predict number of class co-occurrence
        self.reconstruction = nn.Conv2d(8*block.expansion, num_recon_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.dropout = nn.Dropout(p=0.5)

        self._init_weights(load_pretrained=load_pretrained)

    def _init_weights(self, load_pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

        if load_pretrained:
            backbone = model_dict[self.name]['pretrained']
            pretrained_state_dict = backbone.state_dict()
            state_dict = self.state_dict()
            for k in state_dict.keys():
                if 'layer0.0.' in k:
                    v = pretrained_state_dict[k.replace('layer0.0.', 'conv1.')]
                    state_dict[k] = torch.cat((v, v[:, :1, :, :]), dim=1)
                elif 'layer0.1.' in k:
                    state_dict[k] = pretrained_state_dict[k.replace('layer0.1.', 'bn1.')]
                elif 'cbam' not in k and ('layer1.' in k or 'layer2.' in k or 'layer3.' in k or 'layer4.' in k):
                    state_dict[k] = pretrained_state_dict[k]
            self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1, use_cbam=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x, recon_scale=1):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.training:
            aux_x = self.cbr_aux4(x)
            aux_x = F.interpolate(aux_x, scale_factor=2, mode='bilinear', align_corners=False)
            aux_x = self.cbr_aux3(aux_x)
            aux_x = F.interpolate(aux_x, scale_factor=2, mode='bilinear', align_corners=False)
            aux_x = self.cbr_aux2(aux_x)
            aux_x = F.interpolate(aux_x, scale_factor=2, mode='bilinear', align_corners=False)
            recon = self.reconstruction(aux_x)
            if recon_scale > 1:
                recon = F.interpolate(recon, scale_factor=recon_scale, mode='bilinear', align_corners=False)

        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # Global Average Pooling
        x = x.view(x.size(0), -1) # flatten
        x = self.dropout(x)
        cls = self.classification(x)

        if self.training:
            cooc = self.classification_cooc(x)
            return cls, cooc, recon
        else:
            return cls
