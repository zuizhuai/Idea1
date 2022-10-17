import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torchvision.models import resnet50

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()

    def forward(self):
        pass


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = resnet50()
        del self.backbone.fc


    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        print(x.shape)
        b, c, h, w = x.size()

        l_u = x[:, :, 0:int(h / 2), 0:int(w / 2)]
        r_u = x[:, :, 0:int(h / 2), int(w / 2):w]
        l_d = x[:, :, int(h / 2):h, 0:int(w / 2)]
        r_d = x[:, :, int(h / 2):h, int(w / 2):w]
        mid = x[:, :, int(h / 4):int(3 * h / 4), int(w / 4):int(w * 3 / 4)]

        l_u = self.conv(l_u)
        l_d = self.conv(l_d)
        r_u = self.conv(r_u)
        r_d = self.conv(r_d)
        mid = self.conv(mid)
        zeropad = nn.ZeroPad2d(padding=(int(h / 4), int(w / 4), int(h / 4), int(w / 4)))
        mid = zeropad(mid)

        pre_data = torch.cat((torch.cat((l_u, r_u), dim=3), torch.cat((l_d, r_d), dim=3)), dim=2)
        # up_mid = F.upsample(mid, scale_factor=2,mode='bilinear', align_corners=True)
        mid=zeropad(mid)
        print(mid.shape)


if __name__ == '__main__':
    x = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    print(x.shape)
    b, c, h, w = x.size()

    l_u = x[:, :, 0:int(h / 2), 0:int(w / 2)]
    r_u = x[:, :, 0:int(h / 2), int(w / 2):w]
    l_d = x[:, :, int(h / 2):h, 0:int(w / 2)]
    r_d = x[:, :, int(h / 2):h, int(w / 2):w]
    mid = x[:, :, int(h / 4):int(3 * h / 4), int(w / 4):int(w * 3 / 4)]

    pre_data = torch.cat((torch.cat((l_u, r_u), dim=3), torch.cat((l_d, r_d), dim=3)), dim=2)
    print(pre_data)
    print(mid.shape)
    zeropad = nn.ZeroPad2d(padding=(int(h / 4), int(w / 4), int(h / 4), int(w / 4)))
    mid = zeropad(mid)
    print(mid.shape)
    # model = Model()
    # a = torch.rand(size=(1, 3, 224, 224))
    # model(a)
    # summary(model, input_size=(1, 3, 224, 224))
