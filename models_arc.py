import torch
from torch import nn
from torch.nn import functional as F

class ResNetBlock(nn.Module):
    def __init__(self, chn,k=3):
        super().__init__()
        self.l1 = nn.Conv2d(chn, chn*2, k, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(chn*2)
        self.l2 = nn.Conv2d(chn*2, chn, k, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(chn)
        self.act = nn.ReLU()
    def forward(self, x):
        o = self.b1(self.act(self.l1(x)))
        return self.b2(self.act(x + self.l2(o)))


class ConvModel_BnResnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 7, stride=2, bias=False)
        self.res_1 = ResNetBlock(32, 3)
        self.bn1 = nn.BatchNorm2d(32)


        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, bias=False)
        self.res_2 = ResNetBlock(64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, bias=False)
        self.res_3 = ResNetBlock(128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        
        self.conv_4 = nn.Conv2d(128, 256, 3, stride=2, bias=False)
        self.res_4 = ResNetBlock(256, 3)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv = nn.Conv2d(256, 512, 3, stride=2, bias=False)
        
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(512, out_size)
        self.act = nn.ReLU()
    
    def forward(self, x, return_pool=False):
        x = self.bn1(self.act(self.conv_1(x)))
        x = self.res_1(x)

        x = self.bn2(self.act(self.conv_2(x)))
        x = self.res_2(x)

        x = self.bn3(self.act(self.conv_3(x)))
        x = self.res_3(x)

        x = self.bn4(self.act(self.conv_4(x)))
        x = self.res_4(x)

        x = self.act(self.conv(x))
        x = self.pool(x).view(x.shape[0],-1)
        if return_pool:
            return F.log_softmax(self.out(x), dim=-1), x
        return F.log_softmax(self.out(x), dim=-1)
