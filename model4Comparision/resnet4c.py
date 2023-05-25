import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from ._registry import register_model

# _cfg = {
#     'url':'',
#     'num_classes':10, 'input_size':(3,32,32), 'pool_szie':None,
#     'crop_pct':.96, 'interpolation':'bicubic',
#     'mean':IMAGENET_DEFAULT_MEAN, 'std':IMAGENET_DEFAULT_STD, 'classifer':'head'
# }

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


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes,planes,1,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes,planes,3,1,1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.act3 = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self,x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.act3(self.bn3(self.conv3(out)))
        out = out + self.shortcut(x)
        y = F.relu(out)
        return y


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, in_chans, num_classes):
        super(Resnet, self).__init__()

        self.conv1 = nn.Conv2d(in_chans,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False)

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
        # out = self.maxpool(out)
        out = self.features(out)
        out = self.globalpool(out)
        y = self.head(out)

        return y


@register_model
def Resnet18(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[2,2,2,2],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet34(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[3,4,6,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet50(pretrained=False, **kwargs):
    model = Resnet(block=BottleNeck,num_blocks=[3,4,6,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet101(pretrained=False, **kwargs):
    model = Resnet(block=BottleNeck,num_blocks=[3,4,23,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet152(pretrained=False, **kwargs):
    model = Resnet(block=BottleNeck,num_blocks=[3,8,36,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model


# -----------------ResNet series for comparison on 4 datasets------------------------

# -----------------MIN-----------------------
@register_model
def Resnet18_MIN(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[2,2,2,2],in_chans=3,num_classes=100)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet34_MIN(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[3,4,6,3],in_chans=3,num_classes=100)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet50_MIN(pretrained=False, **kwargs):
    model = Resnet(block=BottleNeck,num_blocks=[3,4,6,3],in_chans=3,num_classes=100)
    #model.default_cfgs = _cfg
    return model


# -----------------C10-----------------------
@register_model
def Resnet18_C10(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[2,2,2,2],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet34_C10(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[3,4,6,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet50_C10(pretrained=False, **kwargs):
    model = Resnet(block=BottleNeck,num_blocks=[3,4,6,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model


# -----------------C100-----------------------
@register_model
def Resnet18_C100(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[2,2,2,2],in_chans=3,num_classes=100)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet34_C100(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[3,4,6,3],in_chans=3,num_classes=100)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet50_C100(pretrained=False, **kwargs):
    model = Resnet(block=BottleNeck,num_blocks=[3,4,6,3],in_chans=3,num_classes=100)
    #model.default_cfgs = _cfg
    return model

# -----------------FM-----------------------
@register_model
def Resnet18_FM(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[2,2,2,2],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet34_FM(pretrained=False, **kwargs):
    model = Resnet(block=BasicBlock,num_blocks=[3,4,6,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model

@register_model
def Resnet50_FM(pretrained=False, **kwargs):
    model = Resnet(block=BottleNeck,num_blocks=[3,4,6,3],in_chans=3,num_classes=10)
    #model.default_cfgs = _cfg
    return model







