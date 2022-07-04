# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from utils import batch_index_fill, batch_index_select


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class AdaBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.fast_path = nn.Sequential(
            LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim)
            )
        self.fast_path_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None

    def forward_ffn(self, x):
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x

    def forward(self, x, mask=None):
        input_x = x
        if mask is None: # compatible with the original implementation 
            x = self.dwconv(x)

            x = x.permute(0, 2, 3, 1)        # (N, C, H, W) -> (N, H, W, C)
            x = self.forward_ffn(x)
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

            x = input_x + self.drop_path(x)
            return x
        else:
            if self.training:
                x1, x2 = x
                x0 = x1 * mask + x2 * (1 - mask)
                x = self.dwconv(x0)
                x1 = x * mask + x1 * (1 - mask)
                x2 = x * (1 - mask) + x2 * mask

                x1 = x1.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
                x1 = self.forward_ffn(x1)
                x1 = x1.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

                x2 = x2.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
                x2 = self.fast_path(x2)
                if self.fast_path_gamma is not None:
                    x2 = self.fast_path_gamma * x2
                x2 = x2.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
                
                x1 = input_x[0] + self.drop_path(x1)
                x2 = input_x[1] + self.drop_path(x2)
                x = [x1, x2]
                return x
            else: # inference mode
                idx1, idx2 = mask
                N, C, H, W = x.shape
                x = self.dwconv(x)
                x = x.permute(0, 2, 3, 1).reshape(N, H*W, C) # (N, C, H, W) -> (N, H, W, C)

                x1 = batch_index_select(x, idx1)
                x2 = batch_index_select(x, idx2)
                x1 = self.forward_ffn(x1)
                x2 = self.fast_path(x2)
                if self.fast_path_gamma is not None:
                    x2 = self.fast_path_gamma * x2

                x = torch.zeros_like(x)
                x = batch_index_fill(x, x1, x2, idx1, idx2)

                x = x.reshape(N, H, W, C).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

                x = input_x + self.drop_path(x)
                return x


class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, 2, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_x, mask=None, ratio=0.5):
        if self.training and mask is not None:
            x1, x2 = input_x
            input_x = x1 * mask + x2 * (1 - mask)
        else:
            x1 = input_x
            x2 = input_x
        x = self.in_conv(input_x)
        B, C, H, W = x.size()
        local_x = x[:, :C//2]
        global_x = torch.mean(x[:, C//2:], keepdim=True, dim=(2, 3))
        x = torch.cat([local_x, global_x.expand(B, C//2, H, W)], dim=1)
        pred_score = self.out_conv(x)

        if self.training:
            mask = F.gumbel_softmax(pred_score, hard=True, dim=1)[:, 0:1]
            return [x1, x2], mask
        else:
            score = pred_score[:,0]
            B, H, W = score.shape
            N = H * W
            num_keep_node = int(N * ratio)
            idx = torch.argsort(score.reshape(B, N), dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return input_x, [idx1, idx2]


class AdaConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., sparse_ratio=[0.75, 0.5, 0.25], pruning_loc=[1,2,3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            if i in [0, 1, 3]:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(0, pruning_loc[0])],
                    *[AdaBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(pruning_loc[0], depths[i])]
                )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)


        # new modules

        self.sparse_ratio = sparse_ratio
        predictor_list = [PredictorLG(dims[2]) for i in range(len(pruning_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)

        self.pruning_loc_stage3 = pruning_loc  # [0, 3, 6] for 9 layers

        # init 

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def get_costum_param_groups(self, weight_decay):
        decay = []
        no_decay = []
        new_param = []
        for name, param in self.named_parameters():
            if 'fast_path' in name or 'predictor' in name:
                new_param.append(param)
            elif not param.requires_grad:
                continue  # frozen weights
            elif 'cls_token' in name or 'pos_embed' in name: #or 'patch_embed' in name:
                continue  # frozen weights
            elif len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': new_param, 'name': 'new_param', 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay', 'small_lr_scalar': 0.01, 'fix_step': 5},
            {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay', 'small_lr_scalar': 0.01, 'fix_step': 5}
            ]
    
    def forward(self, x):

        # stage 1, 2
        for i in range(2):
            x = self.downsample_layers[i](x)
            for _, layer in enumerate(self.stages[i]):
                x = layer(x)
        
        # stage 3
        x = self.downsample_layers[2](x)
        pruning_loc = 0
        mask = None

        decisions = []
        for i, layer in enumerate(self.stages[2]):
            if i in self.pruning_loc_stage3:
                x, mask = self.score_predictor[pruning_loc](x, mask, self.sparse_ratio[pruning_loc])
                pruning_loc += 1
                decisions.append(mask)
            if i < self.pruning_loc_stage3[0]:
                x = layer(x)
            else:
                x = layer(x, mask)
        
        # stage 4
        if self.training:
            x1, x2 = x
            x = x1 * mask + x2 * (1 - mask)
        x = self.downsample_layers[3](x)

        for i, layer in enumerate(self.stages[3]):
            x = layer(x)
        
        if self.training:
            featmap = self.norm(x.permute(0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)

        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        
        if self.training:
            return x, featmap, decisions
        else:
            return x

class ConvNeXt_Teacher(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for _, layer in enumerate(self.stages[i]):
                x = layer(x)

        featmap = self.norm(x.permute(0, 2, 3, 1))
        x = self.norm(x.mean([-2, -1]))
        
        x = self.head(x)
        return x, featmap

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x