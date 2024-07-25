import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from ._registry import register_model


class SPC(nn.Module):

    def __init__(self, channels, reduce_rate, out_channels, step, stride=1):
        super().__init__()
        self.step = step
        self.stride = stride
        self.channels = channels
        self.BN = nn.BatchNorm2d(channels)

        self.act = nn.GELU()
        self.fuse_t = nn.Conv2d(channels, channels // reduce_rate, (1, 1))
        self.fuse_b = nn.Conv2d(channels, channels // reduce_rate, (1, 1))
        self.fuse_r = nn.Conv2d(channels, channels // reduce_rate, (1, 1))
        self.fuse_l = nn.Conv2d(channels, channels // reduce_rate, (1, 1))
        self.fuse1 = nn.Conv2d((channels//reduce_rate) * 4, out_channels, (1, 1))
        # self.fuse2 = nn.Conv2d(channels * 3, out_channels, (1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.act(self.BN(x))
        x_t, x_b, x_r, x_l = x.clone(), x.clone(), x.clone(), x.clone()
        x_t = torch.narrow(x_t, 2, self.step, (H-self.step))
        x_b = torch.narrow(x_b, 2, 0, (H-self.step))
        x_l = torch.narrow(x_l, 3, self.step, (W-self.step))
        x_r = torch.narrow(x_r, 3, 0, (W-self.step))
        x_t = F.pad(x_t, (0, 0, 0, self.step), "constant", 0)
        x_b = F.pad(x_b, (0, 0, self.step, 0), "constant", 0)
        x_l = F.pad(x_l, (0, self.step, 0, 0), "constant", 0)
        x_r = F.pad(x_r, (self.step, 0, 0, 0), "constant", 0)

        x_t = self.fuse_t(x_t)
        x_b = self.fuse_b(x_b)
        x_r = self.fuse_r(x_r)
        x_l = self.fuse_l(x_l)

        x = self.fuse1(torch.cat([x_t, x_b, x_r, x_l], dim=1))
        if self.stride != 1:
            x = self.maxpool(x)
        return x


class BasicBlock(nn.Module):
    expansion =1
    def __init__(self,in_planes,planes,stride):
        super(BasicBlock, self).__init__()
        #self.expansion = 1

        self.conv1 = nn.Conv2d(in_planes,planes,3,stride,1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes,planes,3,1,1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=False)

        self.residual = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.residual = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,1,stride,bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self,x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = out + self.residual(x)

        return out


class BasicBlock_spc(nn.Module):
    expansion =1
    def __init__(self,in_planes,planes,stride):
        super(BasicBlock, self).__init__()
        #self.expansion = 1

        # self.conv1 = nn.Conv2d(in_planes,planes,3,stride,1)
        self.conv1 = SPC(in_planes, 4, planes, 1, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=False)
        # self.conv2 = nn.Conv2d(planes,planes,3,1,1)
        self.conv2 = SPC(planes, 4, planes, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=False)

        self.residual = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.residual = nn.Sequential(
                # nn.Conv2d(in_planes,self.expansion*planes,1,stride,bias=False),
                SPC(in_planes, 4, self.expansion*planes, step=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self,x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = out + self.residual(x)

        return out


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, in_chans, num_classes):
        super(Resnet, self).__init__()

        self.conv1 = nn.Conv2d(in_chans,64,kernel_size=7,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=False)

        self.in_planes = 64
        self.features = nn.Sequential(
            self._make_layers(block, num_blocks[0], 64, 1),
            self._make_layers(block, num_blocks[1], 128, 2),
            self._make_layers(block, num_blocks[2], 256, 2),
            self._make_layers(block, num_blocks[3], 512, 2)
        )

        self.globalpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1)
        )

        self.head = nn.Linear(512*block.expansion,num_classes,bias=True)

    def _make_layers(self,block, num_blocks, planes, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.features(out)
        out = self.globalpool(out)
        y = self.head(out)

        return y


class Resnet_spc64(nn.Module):
    def __init__(self, block, num_blocks, in_chans, num_classes):
        super(Resnet, self).__init__()

        self.conv1 = nn.Conv2d(in_chans,64,kernel_size=7,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=False)


        self.in_planes = 64
        self.features = nn.Sequential(
            self._make_layers(block, num_blocks[0], 64, 1),
            self._make_layers(block, num_blocks[1], 128, 2),
            self._make_layers(block, num_blocks[2], 256, 2),
            self._make_layers(block, num_blocks[3], 512, 2)
        )

        self.globalpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1)
        )

        self.head = nn.Linear(512*block.expansion,num_classes,bias=True)

    def _make_layers(self,block, num_blocks, planes, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.features(out)
        out = self.globalpool(out)
        y = self.head(out)

        return y


class Resnet_spc96(nn.Module):
    def __init__(self, block, num_blocks, in_chans, num_classes):
        super(Resnet, self).__init__()

        self.conv1 = nn.Conv2d(in_chans,96,kernel_size=7,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(96)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=False)


        self.in_planes = 96
        self.features = nn.Sequential(
            self._make_layers(block, num_blocks[0], 96, 1),
            self._make_layers(block, num_blocks[1], 192, 2),
            self._make_layers(block, num_blocks[2], 384, 2),
            self._make_layers(block, num_blocks[3], 768, 2)
        )

        self.globalpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1)
        )

        self.head = nn.Linear(768*block.expansion,num_classes,bias=True)

    def _make_layers(self,block, num_blocks, planes, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.features(out)
        out = self.globalpool(out)
        y = self.head(out)

        return y


class Resnet_spc128(nn.Module):
    def __init__(self, block, num_blocks, in_chans, num_classes):
        super(Resnet, self).__init__()

        self.conv1 = nn.Conv2d(in_chans,128,kernel_size=7,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=False)


        self.in_planes = 128
        self.features = nn.Sequential(
            self._make_layers(block, num_blocks[0], 128, 1),
            self._make_layers(block, num_blocks[1], 256, 2),
            self._make_layers(block, num_blocks[2], 512, 2),
            self._make_layers(block, num_blocks[3], 1024, 2)
        )

        self.globalpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1)
        )

        self.head = nn.Linear(1024*block.expansion,num_classes,bias=True)

    def _make_layers(self,block, num_blocks, planes, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.features(out)
        out = self.globalpool(out)
        y = self.head(out)

        return y


@register_model
def Resnet18(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_chans=3, num_classes=1000)
    # model.default_cfgs = _cfg
    return model

@register_model
def Resnet18_spc64(pretrained=False, **kwargs):
    model = Resnet_spc64(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_chans=3, num_classes=1000)
    # model.default_cfgs = _cfg
    return model

@register_model
def Resnet18_spc96(pretrained=False, **kwargs):
    model = Resnet_spc96(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_chans=3, num_classes=1000)
    # model.default_cfgs = _cfg
    return model

@register_model
def Resnet18_spc128(pretrained=False, **kwargs):
    model = Resnet_spc128(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_chans=3, num_classes=1000)
    # model.default_cfgs = _cfg
    return model
