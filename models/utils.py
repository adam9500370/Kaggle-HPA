import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, groups=1, is_batchnorm=True, momentum=0.1):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters), momentum=momentum),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, groups=1, negative_slope=0.0, is_batchnorm=True, momentum=0.1):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)

        if negative_slope > 0.0:
            relu_mod = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            relu_mod = nn.ReLU(inplace=True)

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters), momentum=momentum),
                                          relu_mod,)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          relu_mod,)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class pyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, is_batchnorm=True):
        super(pyramidPooling, self).__init__()

        bias = not is_batchnorm

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, is_batchnorm=is_batchnorm))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        h, w = x.shape[2:]

        output_slices = [x]
        for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
            stride = (int(h/pool_size), int(w/pool_size))
            k_size = (int(h - stride[0]*(pool_size-1)), int(w - stride[1]*(pool_size-1)))
            out = F.avg_pool2d(x, k_size, stride=stride, padding=0)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, 
                 stride, dilation=1, groups=1, is_batchnorm=True, use_cbam=False):
        super(bottleNeckPSP, self).__init__()

        bias = not is_batchnorm

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cbam = use_cbam

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                        stride=stride, padding=dilation,
                                        bias=bias, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        if self.in_channels != self.out_channels:
            self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride=stride, padding=0, bias=bias, is_batchnorm=is_batchnorm)

        if self.use_cbam:
            self.cbam = CBAM(gate_channels=out_channels)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        if self.use_cbam:
            conv = self.cbam(conv)
        residual = self.cb4(x) if self.in_channels != self.out_channels else x
        return F.relu(conv+residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, groups=1, is_batchnorm=True, use_cbam=False):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not is_batchnorm

        self.use_cbam = use_cbam

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                        stride=1, padding=dilation,
                                        bias=bias, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)

        if self.use_cbam:
            self.cbam = CBAM(gate_channels=in_channels)
        
    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        if self.use_cbam:
            x = self.cbam(x)
        return F.relu(x+residual, inplace=True)


class residualBlockPSP(nn.Module):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1, groups=1, include_range='all', is_batchnorm=True, use_cbam=False):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        # residualBlockPSP = convBlockPSP + identityBlockPSPs
        layers = []
        if include_range in ['all', 'conv']:
            layers.append(bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm, use_cbam=use_cbam))
        if include_range in ['all', 'identity']:
            for i in range(n_blocks-1):
                layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm, use_cbam=use_cbam))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



# CBAM: https://github.com/Jongchan/attention-module
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()

        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type == 'max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()

        self.compress = ChannelPool()
        self.spatial = conv2DBatchNorm(in_channels=2, n_filters=1, k_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False, is_batchnorm=True, momentum=0.01)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
