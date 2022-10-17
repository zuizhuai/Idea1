import torch.nn as nn
import torch
import torch.nn.functional as F


def get_pad_layer(pad_type):
    global PadLayer
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)
        # (n,gxkxk,h,w)
        n, c, h, w = sigma.shape
        # (n,1,gxkxk,hxw)
        sigma = sigma.reshape(n, 1, c, h * w)

        n, c, h, w = x.shape
        # (n,cxkxk,hxw) -> (n,c,kxk,hxw)
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            (n, c, self.kernel_size * self.kernel_size, h * w))

        n, c1, p, q = x.shape
        # (n,c,kxk,hxw) -> (n,g,c/g,kxk,hxw)
        x = x.permute(1, 0, 2, 3).reshape(self.group, c1 // self.group, n, p, q).permute(2, 0, 1, 3, 4)

        n, c2, p, q = sigma.shape
        # (gxkxk,n,1,hxw) -> (g,kxk,n,1,hxw) ->(n,g,1,kxk,hxw)
        sigma = sigma.permute(2, 0, 1, 3).reshape(
            (p // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0,
                                                                                                                 3, 1,
                                                                                                                 4)

        temp1=torch.sum(x * sigma, dim=3)
        # sigma(n,g,1,kxk,hxw) * x(n,g,c/g,kxk,hxw) -> (n,g,c/g,hxw)
        x = torch.sum(x * sigma, dim=3).reshape(n, c1, h, w)
        return x[:, :, torch.arange(h) % self.stride == 0, :][:, :, :, torch.arange(w) % self.stride == 0]


if __name__=='__main__':
    input=torch.rand(size=(1,4,14,14))
    model=Downsample_PASA_group_softmax(in_channels=4,kernel_size=3)
    out=model(input)