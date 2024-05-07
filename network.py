import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
import sys



class res2net_3Dconv(nn.Module):

    def __init__(self, inp, S=4, kernel=(1, 1, 7)):
        super(res2net_3Dconv, self).__init__()

        if inp % S != 0:
            raise ValueError('Planes must be divisible by scales')

        self.S = S
        part_channel = int(inp / S)

        a = int((kernel[0] - 1) / 2)
        b = int((kernel[1] - 1) / 2)
        c = int((kernel[2] - 1) / 2)

        self.conv1 = nn.Conv3d(inp, inp, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(inp)

        self.conv2 = nn.ModuleList([nn.Conv3d(part_channel, part_channel,
                                              kernel_size=kernel, stride=1, padding=(a, b, c)) for _ in range(S-1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(part_channel) for _ in range(S-1)])

        self.conv3 = nn.Conv3d(inp, inp, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm3d(inp)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        # X = self.relu(self.bn1(self.conv1(X)))

        xs = torch.chunk(X, self.S, 1)
        ys = []
        for s in range(self.S):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)
        out = self.relu(self.bn3(self.conv3(out)))

        return out
class res2net_2Dconv(nn.Module):

    def __init__(self, inp, S=4, kernel=(3, 3)):
        super(res2net_2Dconv, self).__init__()

        if inp % S != 0:
            raise ValueError('Planes must be divisible by scales')

        self.S = S
        part_channel = int(inp / S)

        a = int((kernel[0] - 1) / 2)
        b = int((kernel[1] - 1) / 2)

        self.conv1 = nn.Conv2d(inp, inp, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(inp)

        self.conv2 = nn.ModuleList([nn.Conv2d(part_channel, part_channel,
                                              kernel_size=kernel, stride=1, padding=(a, b)) for _ in range(S-1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(part_channel) for _ in range(S-1)])

        self.conv3 = nn.Conv2d(inp, inp, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(inp)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        # X = self.relu(self.bn1(self.conv1(X)))

        xs = torch.chunk(X, self.S, 1)
        ys = []
        for s in range(self.S):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)
        out = self.relu(self.bn3(self.conv3(out)))

        return out
class SKConv_channel(nn.Module):
    def __init__(self, inp, oup):
        super(SKConv_channel, self).__init__()
        self.inp = inp
        self.oup = oup

        self.conv1 = nn.Conv3d(in_channels=self.inp, out_channels=self.oup, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1), dilation=(1, 1, 1))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(self.oup, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=False)
        )

        self.conv2 = nn.Conv3d(in_channels=self.inp, out_channels=self.oup, padding=(0, 0, 15),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1), dilation=(1, 1, 5))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(self.oup, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=False)
        )

        self.conv3 = nn.Conv3d(in_channels=self.inp, out_channels=self.oup, padding=(0, 0, 27),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1), dilation=(1, 1, 9))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(self.oup, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=False)
        )

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv2d(self.oup, self.oup, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(self.oup, self.oup, kernel_size=1, stride=1, padding=0)
        self.fc3 = nn.Conv2d(self.oup, self.oup, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x11 = self.batch_norm1(self.conv1(x))
        x12 = self.batch_norm2(self.conv2(x))
        x13 = self.batch_norm3(self.conv3(x))

        x2 = x11 + x12 + x13
        x3 = self.gap(x2).squeeze(-1)

        x41 = self.fc1(x3).view(x3.size(0), 1, 1, x3.size(1))
        x42 = self.fc2(x3).view(x3.size(0), 1, 1, x3.size(1))
        x43 = self.fc3(x3).view(x3.size(0), 1, 1, x3.size(1))

        x4 = torch.cat([x41, x42, x43], dim=1)

        x4 = x4.permute(0, 3, 1, 2)
        x4 = self.softmax(x4)
        x5 = x4.permute(0, 2, 3, 1)
        x51, x52, x53 = torch.split(x5, 1, dim=1)

        x51 = x51.permute(0, 3, 1, 2).unsqueeze(-1)
        x52 = x52.permute(0, 3, 1, 2).unsqueeze(-1)
        x53 = x53.permute(0, 3, 1, 2).unsqueeze(-1)

        out = x11*x51 + x12*x52 + x13*x53
        return out
class SKConv_space(nn.Module):
    def __init__(self, inp, oup):
        super(SKConv_space, self).__init__()
        self.inp = inp
        self.oup = oup

        self.conv1 = nn.Conv2d(in_channels=self.inp, out_channels=self.oup, padding=1,
                               kernel_size=3, stride=1, dilation=1)
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm2d(self.oup, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Conv2d(in_channels=self.inp, out_channels=self.oup, padding=2,
                               kernel_size=3, stride=1, dilation=2)
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm2d(self.oup, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Conv2d(in_channels=self.inp, out_channels=self.oup, padding=3,
                               kernel_size=3, stride=1, dilation=3)
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm2d(self.oup, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=False)
        )

        self.fc1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x11 = self.batch_norm1(self.conv1(x))
        x12 = self.batch_norm2(self.conv2(x))
        x13 = self.batch_norm3(self.conv3(x))

        x2 = x11 + x12 + x13
        x3 = torch.mean(x2, dim=1, keepdim=True)

        x41 = self.fc1(x3).view(x3.size(0), 1, 1, x3.size(2)*x3.size(3))
        x42 = self.fc2(x3).view(x3.size(0), 1, 1, x3.size(2)*x3.size(3))
        x43 = self.fc3(x3).view(x3.size(0), 1, 1, x3.size(2)*x3.size(3))
        x4 = torch.cat([x41, x42, x43], dim=1)
        x4 = x4.permute(0, 3, 1, 2)
        x4 = self.softmax(x4)
        x5 = x4.permute(0, 2, 3, 1)
        x51, x52, x53 = torch.split(x5, 1, dim=1)
        x51 = x51.view(x3.size(0), 1, x3.size(2), x3.size(3))
        x52 = x51.view(x3.size(0), 1, x3.size(2), x3.size(3))
        x53 = x51.view(x3.size(0), 1, x3.size(2), x3.size(3))

        out = x11*x51 + x12*x52 + x13*x53
        return out
class GCAttention_channel(nn.Module):

    _abbr_ = 'GCAttention_channel'

    def __init__(self,
                 in_channels: int,
                 patch_size: 4,
                 ratio: int,
                 pooling_type: str = 'att',
                 fusion_types: tuple = ('channel_add', )):
        super().__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channels = in_channels
        self.ratio = 1 / ratio
        self.planes = int(in_channels * self.ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm(normalized_shape=[self.planes, 1, 1], eps=0.001, elementwise_affine=True),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm(normalized_shape=[self.planes, 1, 1], eps=0.001, elementwise_affine=True),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        else:
            self.channel_mul_conv = None
        # self.reset_parameters()

        self.gamma = nn.Parameter(torch.zeros(1))

    # def reset_parameters(self):
    #     if self.pooling_type == 'att':
    #         kaiming_init(self.conv_mask, mode='fan_in')
    #         self.conv_mask.inited = True
    #
    #     if self.channel_add_conv is not None:
    #         last_zero_init(self.channel_add_conv)
    #     if self.channel_mul_conv is not None:
    #         last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # identity = x
        out = x
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = self.gamma * channel_mul_term * out

        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = self.gamma * channel_add_term + out

        return out
class GCAttention_space(nn.Module):

    _abbr_ = 'GCAttention_space'

    def __init__(self, in_channels: int, patch_size: 4, ratio: int):

        super().__init__()

        self.in_channels = in_channels
        self.ratio = 1 / ratio
        self.patch_size = patch_size * 2 + 1
        self.planes = int(self.patch_size * self.patch_size * self.ratio)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.patch_size * self.patch_size, self.planes, kernel_size=1),
            nn.LayerNorm(normalized_shape=[self.planes, 1, 1],  eps=0.001, elementwise_affine=True),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.patch_size * self.patch_size, kernel_size=1))

        self.gamma = nn.Parameter(torch.zeros(1))

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C , H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, H * W, C]
        input_x = input_x.permute(0, 2, 1)
        # [N, 1, H * W, C]
        input_x = input_x.unsqueeze(1)
        # [N, C, 1, 1]
        context_mask = self.GAP(x)
        # [N, 1, C]
        context_mask = context_mask.view(batch, 1, channel)
        # [N, 1, C]
        context_mask = self.softmax(context_mask)
        # [N, 1, C, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, H * W, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, H * W, 1, 1]
        context = context.view(batch, height * width, 1, 1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        identity = x
        # [N, H * W, 1, 1]
        context = self.spatial_pool(x)

        channel_add_term = (self.channel_add_conv(context)).reshape(batch, 1, height, width)
        out = self.gamma * channel_add_term + identity

        return out



class res2_SK_GAatt(nn.Module):
    # 在空间和光谱上只使用一个res2net，在res2net内部的最后没有使用卷积融合不同部分的特征图。没有SK，没有注意力
    def __init__(self, band, classes):
        super(res2_SK_GAatt, self).__init__()

        # spectral branch
        self.name = 'res2_SK_GAatt'

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=60,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # 3D res2net
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.res2net_3Dconv1 = res2net_3Dconv(60, 4, kernel=(1, 1, 7))
        # self.res2net_3Dconv2 = res2net_3Dconv(24, 4, kernel=(1, 1, 7))
        # self.res2net_3Dconv3 = res2net_3Dconv(24, 4, kernel=(1, 1, 7))

        kernel_3d = math.ceil((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        self.batch_norm15 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )


        # Spatial Branch
        self.conv21 = nn.Conv2d(in_channels=band, out_channels=60, kernel_size=3, stride=1, padding=1)
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3, stride=1, padding=1)

        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.res2net_2Dconv1 = res2net_2Dconv(60, 4, kernel=(3, 3))
        # self.res2net_2Dconv2 = res2net_2Dconv(24, 4, kernel=(3, 3))
        # self.res2net_2Dconv3 = res2net_2Dconv(24, 4, kernel=(3, 3))

        self.conv25 = nn.Conv2d(in_channels=60, out_channels=60, padding=0,
                                kernel_size=1, stride=1)

        self.batch_norm25 = nn.Sequential(
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        # 注意力
        self.GCAttention_channel = GCAttention_channel(60, 4, 16)
        self.GCAttention_space = GCAttention_space(60, 4, 16)

        self.SKConv_channel = SKConv_channel(60, 60)        #在attontion.py文件里
        self.SKConv_space = SKConv_space(60, 60)

        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.full_connection = nn.Sequential(
            nn.Linear(120, classes)
        )

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x11 = self.batch_norm11(x11)
        x11_SK = self.SKConv_channel(x11)

        x12 = self.res2net_3Dconv1(x11)

        x15 = self.conv15(x11 + x12 + x11_SK)

        # # 附加
        # x15 = self.batch_norm15(x15)
        # # 附加

        # 光谱注意力机制
        x1 = self.batch_norm1(self.GCAttention_channel(x15.squeeze(-1)))
        x1 = self.global_pooling(x1)

        x1 = x1.squeeze(-1).squeeze(-1)

        # spatial
        # 空间分支  特征图降维  变为（32,200,9,9）
        x = X.permute(0, 4, 2, 3, 1)
        x = x.squeeze(-1)

        x21 = self.conv21(x)
        x21 = self.batch_norm21(x21)
        # x22 = self.conv22(x21)
        # x22 = self.batch_norm22(x22)

        x21_SK = self.SKConv_space(x21)

        x22 = self.res2net_2Dconv1(x21)

        # 附加
        x25 = self.conv25(x21 + x22 + x21_SK)
        # x25 = self.batch_norm25(x25)
        # 附加

        # 空间注意力机制
        x2 = self.batch_norm2(self.GCAttention_space(x25))
        x2 = self.global_pooling(x2)
        # print(x2.shape)
        x2 = x2.squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        output = self.full_connection(x_pre)
        return output

