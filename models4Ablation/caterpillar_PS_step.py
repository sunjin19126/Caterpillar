import torch
from torch import nn
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp
from ._registry import register_model
from timm.models.layers import DropPath, PatchEmbed


class ShiftedPillarsConcatentation_BNAct(nn.Module):

    def __init__(self, channels, step):
        super().__init__()
        self.channels = channels
        self.step = step
        self.BN = nn.BatchNorm2d(channels)
        self.Act = nn.GELU()

        self.proj_t = nn.Conv2d(channels, channels // 4, (1, 1))
        self.proj_b = nn.Conv2d(channels, channels // 4, (1, 1))
        self.proj_l = nn.Conv2d(channels, channels // 4, (1, 1))
        self.proj_r = nn.Conv2d(channels, channels // 4, (1, 1))
        self.fuse = nn.Conv2d(channels, channels, (1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.Act(self.BN(x))

        x_t, x_b, x_r, x_l = x.clone(), x.clone(), x.clone(), x.clone()
        x_t = self.proj_t(x_t)
        x_b = self.proj_b(x_b)
        x_l = self.proj_l(x_l)
        x_r = self.proj_r(x_r)

        x_t = F.pad(x_t, (0, 0, 0, self.step), "constant", 0)
        x_b = F.pad(x_b, (0, 0, self.step, 0), "constant", 0)
        x_l = F.pad(x_l, (0, self.step, 0, 0), "constant", 0)
        x_r = F.pad(x_r, (self.step, 0, 0, 0), "constant", 0)
        x_t = torch.narrow(x_t, 2, self.step, H)
        x_b = torch.narrow(x_b, 2, 0, H)
        x_l = torch.narrow(x_l, 3, self.step, W)
        x_r = torch.narrow(x_r, 3, 0, W)

        x = self.fuse(torch.cat([x_t, x_b, x_r, x_l], dim=1))
        return x


class sparseMLP_BNAct(nn.Module):

    def __init__(self, W, H, channels):
        super().__init__()
        assert W == H
        self.channels = channels
        self.BN = nn.BatchNorm2d(channels)
        self.Act = nn.GELU()

        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proj_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.Act(self.BN(x))

        x_o, x_h, x_w = x.clone(), x.clone(), x.clone()
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proj_w(x_w.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x_o, x_h, x_w], dim=1))
        return x


class FeedForward_LN(nn.Module):

    def __init__(self, channels, drop_out=0.):
        super().__init__()
        self.LN = nn.LayerNorm(channels)
        self.FeedForward = Mlp(channels, channels * 3, channels, drop=drop_out)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)
        x = self.LN(x)
        x = self.FeedForward(x)
        x = x.permute(0,3,1,2)
        return x


class CaterpillarBlock_A2_2(nn.Module):

    def __init__(self, W, H, channels, step, drop_out, drop_path):
        super().__init__()
        self.localMix = ShiftedPillarsConcatentation_BNAct(channels, step)
        self.globalMix = sparseMLP_BNAct(W, H, channels)
        self.channelMix = FeedForward_LN(channels,drop_out)

        self.drop_path_g = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_c = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path_g(self.globalMix(self.localMix(x))) + x
        x = self.drop_path_c(self.channelMix(x)) + x
        return x


class Caterpillar_A2_2(nn.Module):
    def __init__(self, img_size=224, in_chans=3, patch_size=4, embed_dim=[96,192,384,768], depth=[2,6,14,2],
                 down_stride=[2,2,2,2], shift_step=[1,1,1,1],
                 num_classes=1000, drop_rate=0., drop_path_rate=0.0, Patch_layer=PatchEmbed, act_layer=nn.GELU):
        super().__init__()
        assert patch_size == down_stride[0], "The down_sampling_stride[0] must be equal to the patch size."
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]


        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim[0], flatten=False, bias=False)
        num_patch = self.patch_embed.grid_size[0]
        dpr = [drop_path_rate * j / (sum(depth)) for j in range(sum(depth))]

        self.blocks = nn.ModuleList([])
        shift = 0
        for i in range(len(depth)):
            if i == 0:
                self.blocks.append(nn.Identity())
            else:
                num_patch = num_patch // down_stride[i]
                self.blocks.append(nn.Conv2d(embed_dim[i-1], embed_dim[i], down_stride[i], down_stride[i], bias=False))

            for j in range(depth[i]):
                drop_path=dpr[j+shift]
                self.blocks.append(nn.Sequential(CaterpillarBlock_A2_2(num_patch, num_patch, embed_dim[i], shift_step[i],
                                                                  drop_out=drop_rate, drop_path=dpr[j+shift])))
            shift += depth[i]
            self.blocks = nn.Sequential(*self.blocks)

        self.norm = nn.BatchNorm2d(embed_dim[-1])


        self.feature_info = [dict(num_chs=embed_dim[-1], reduction=0, module='head')]
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x).mean(dim=[2,3]).flatten(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



# 0000
@register_model
def Caterpillar_A2_2_T_MIN_0000(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=84, patch_size=3,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[3,2,2,1],
                        shift_step=[0,0,0,0], **kwargs)
    return model

@register_model
def Caterpillar_A2_2_T_C10_0000(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=32, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[0,0,0,0], **kwargs)
    return model

@register_model
def Caterpillar_A2_2_T_C100_0000(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=32, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[0,0,0,0], **kwargs)
    return model

@register_model
def Caterpillar_A2_2_T_FM_0000(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=28, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,1],
                        shift_step=[0,0,0,0], **kwargs)
    return model

# --------------------------------------------------------------
# 2222
@register_model
def Caterpillar_A2_2_T_MIN_2222(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=84, patch_size=3,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[3,2,2,1],
                        shift_step=[2,2,2,2], **kwargs)
    return model

@register_model
def Caterpillar_A2_2_T_C10_2222(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=32, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[2,2,2,2], **kwargs)
    return model

@register_model
def Caterpillar_A2_2_T_C100_2222(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=32, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,2],
                        shift_step=[2,2,2,2], **kwargs)
    return model

@register_model
def Caterpillar_A2_2_T_FM_2222(pretrained=False, **kwargs):
    model = Caterpillar_A2_2(img_size=28, patch_size=1,
                        embed_dim=[80,160,320,640],
                        depth=[2,8,14,2],
                        down_stride=[1,2,2,1],
                        shift_step=[2,2,2,2], **kwargs)
    return model

