import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from utils import Anti_aliasing
from torchinfo import summary


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


# x_dim=[64,256,512,1024,2048]
# 目的：降维，融合并消除相似通道
# class channel_agg(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(channel_agg, self).__init__()
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.c_conv1 = nn.Sequential(nn.Conv2d(in_dim[0], out_dim[0], kernel_size=3, stride=1, padding=1,
#                                                groups=out_dim[0]),
#                                      nn.BatchNorm2d(out_dim[0]),
#                                      nn.ReLU(inplace=True))
#         self.c_conv2 = nn.Conv2d(in_dim[1], out_dim[1], kernel_size=3, stride=1, padding=1,
#                                  groups=out_dim[1])
#         self.c_conv3 = nn.Conv2d(in_dim[2], out_dim[2], kernel_size=3, stride=1, padding=1,
#                                  groups=out_dim[2])
#
#         self.conv = nn.Conv2d(sum(in_dim), out_dim[1], 3, 1, 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.beta = nn.Parameter(torch.ones(size=(1,)), requires_grad=True)
#         self.alpha = nn.Parameter(torch.ones(size=(1,)), requires_grad=True)
#
#     def forward(self, x1, x2, x3):
#         x1 = self.max_pool(x1) + self.avg_pool(x1)
#         x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
#         c1 = self.c_conv(x1)
#         c2 = self.c_conv2(x2)
#         c3 = self.c_conv3(x3)
#         c4 = self.conv(torch.cat([x1, x2, x3], dim=1))
#
#         return c4
class pool_bi(nn.Module):
    def __init__(self):
        super(pool_bi, self).__init__()
        self.max_p = nn.MaxPool2d(2, 2)
        self.avg_p = nn.AvgPool2d(2, 2)

    def forward(self, high_f, mid_f, low_f, is_fist=False):
        if is_fist:
            low_f = F.interpolate(low_f, scale_factor=2, mode='bilinear')
            return high_f,mid_f,low_f
        else:
            high_f = self.max_p(high_f) + self.avg_p(high_f)
            low_f = F.interpolate(low_f, scale_factor=2, mode='bilinear')
        return high_f,mid_f,low_f


class c_agg(nn.Module):
    def __init__(self, in_c, out_c):
        super(c_agg, self).__init__()
        self.low = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.mid = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.high = nn.Sequential(
            nn.Conv2d(in_c[2], out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 5, 1, 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, feature):
        low,mid,high = feature
        low = self.low(low)
        mid = self.mid(mid)
        high = self.high(high)
        agg1 = self.conv1(torch.cat((low, mid), dim=1))
        agg1 = agg1 * mid + low

        agg1 = self.mid(agg1)

        agg2 = self.conv2(torch.cat((agg1, high), dim=1))
        agg2 = agg2 * high + agg1

        return agg2


class g_agg(nn.Module):
    def __init__(self, in_c, out_c):
        super(g_agg, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.c_attn = ChannelAttention(out_c)
        self.s_attn = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        w1 = self.c_attn(x)
        w2 = self.s_attn(x * w1)

        return w2 * (w1 * x)


class h_l_agg(nn.Module):
    def __init__(self, h_c,l_c, o_c, factor):
        super(h_l_agg, self).__init__()
        self.factor = factor
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(h_c, l_c, 3, 1, 1),
            nn.BatchNorm2d(l_c),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(l_c * 2, o_c, 1, 1),
            nn.BatchNorm2d(o_c),
            nn.ReLU(),
            nn.Conv2d(o_c, o_c, 3, 1, 1),
            nn.BatchNorm2d(o_c),
            nn.ReLU()
        )

        self.edge = nn.Sequential(
            nn.Conv2d(o_c, o_c, 3, 1, 1),
            nn.BatchNorm2d(o_c),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(l_c * 3, o_c, 1, 1),
            nn.BatchNorm2d(o_c),
            nn.ReLU(),
            nn.Conv2d(o_c, o_c, 3, 1, 1),
            nn.BatchNorm2d(o_c),
            nn.ReLU(),
            nn.Conv2d(o_c, o_c, 1, 1),
            nn.BatchNorm2d(o_c),
            nn.ReLU()
        )

    # edge 作为边界损失而已
    def forward(self, x, y, revers=None):
        y = self.conv1(y)
        y = F.interpolate(y, scale_factor=self.factor, mode='bilinear')
        if revers is not None:
            out = self.conv2(torch.cat((x, y, revers), dim=1))
            out_ = self.conv2(torch.cat((1 - x, 1 - y, revers), dim=1))
        else:
            out = self.conv(torch.cat((x, y), dim=1))
            out_ = self.conv(torch.cat((1 - x, 1 - y), dim=1))
        out = x * y + out * self.sig(y)
        out_ = (1 - x) * (1 - y) + out_ * self.sig(1 - y)
        edge = self.edge(out - out_)
        return edge, out_, out


class Model(nn.Module):
    def __init__(self, g=8,backbone_pre=True):
        super(Model, self).__init__()
        self.g = g
        self.res = resnet50(pretrained=backbone_pre)
        self.c_g1 = c_agg([64 // self.g, 256 // self.g, 512 // self.g], 256 // self.g)
        self.c_g2 = c_agg([256 // self.g, 512 // self.g, 1024 // self.g], 512 // self.g)
        self.c_g3 = c_agg([512 // self.g, 1024 // self.g, 2048 // self.g], 1024 // self.g)
        self.pool_b = pool_bi()

        self.g1 = g_agg(256, 256)
        self.g2 = g_agg(512, 512)
        self.g3 = g_agg(1024, 1024)

        self.filter = Anti_aliasing.Downsample_PASA_group_softmax(in_channels=64, kernel_size=3)

        self.h_l_1 = h_l_agg(1024, 64, 64, factor=4)
        self.h_l_2 = h_l_agg(512, 64, 64, factor=2)
        self.h_l_3 = h_l_agg(256, 64, 64, factor=1)

    def forward(self, x):
        x = self.res.conv1(x)
        x = self.res.bn1(x)
        x1 = self.res.relu(x)  # (b,64,h/2,w/2)
        x2 = self.res.layer1(x1)  # (b,256,h/4,w/4)
        x3 = self.res.layer2(x2)  # (b,512,h/8,w/8)
        x4 = self.res.layer3(x3)  # (b,1024,h/16,w/16)
        x5 = self.res.layer4(x4)  # (b,2048,h/32,w/32)
        print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        x_1 = torch.split(x1, x1.data.size(1) // self.g, dim=1)
        x_2 = torch.split(x2, x2.data.size(1) // self.g, dim=1)
        x_3 = torch.split(x3, x3.data.size(1) // self.g, dim=1)
        x_4 = torch.split(x4, x4.data.size(1) // self.g, dim=1)
        x_5 = torch.split(x5, x5.data.size(1) // self.g, dim=1)
        out_1 = []
        out_2 = []
        out_3 = []
        for i in range(self.g):
            out_1.append(self.c_g1(self.pool_b(x_1[i], x_2[i], x_3[i], is_fist=True)))
            out_2.append(self.c_g2(self.pool_b(x_2[i], x_3[i], x_4[i])))
            out_3.append(self.c_g3(self.pool_b(x_3[i], x_4[i], x_5[i])))

        out1 = channel_shuffle(torch.cat(out_1, dim=1), groups=self.g)
        out2 = channel_shuffle(torch.cat(out_2, dim=1), groups=self.g)
        out3 = channel_shuffle(torch.cat(out_3, dim=1), groups=self.g)

        out1 = self.g1(out1)  # (b,256,h/8,w/8)
        out2 = self.g2(out2)  # (b,512,h/8,w/8)
        out3 = self.g3(out3)  # (b,1024,h/16,w/16)

        temp = self.filter(x1)
        edge1, out_, out = self.h_l_1(temp, out3)
        edge1, out_, out = self.h_l_2(out, out2, out_)
        edge1, out_, out = self.h_l_3(out, out1, out_)

        return F.interpolate(out, scale_factor=2, mode='bilinear')


if __name__ == '__main__':
    input = torch.rand(size=(1, 3, 352, 352)).cuda()
    model = Model().cuda()
    summary(model,input_size=(1,3,352,352))
    out = model(input)
    print(out.shape)
