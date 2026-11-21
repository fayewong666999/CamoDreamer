import math
import numpy as np
import torch

import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
from einops.layers.torch import Rearrange
from camorender.layers import PositionNet
import torchvision.ops as ops

    
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from pycocotools import mask as maskUtils
from .modules import FrozenCLIPImageEmbedder

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
      
        self.pos_embedding = nn.Parameter(torch.empty(1, 256, 768).normal_(std=0.02))
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        feature_scales = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_scales.append(x)
        return feature_scales[-1]

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = x + self.pos_embedding
        return x

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


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(depths=[3, 3, 9, 3],pretrained=True,in_22k=False, **kwargs):
    model = ConvNeXt(depths=depths, dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.load("/cpfs/user/wenzhuangwang/TIP/convnext_tiny_22k_224.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False) # we remove classifer head 
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads


        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)


        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )


    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim 


    def forward(self, x, key, value, mask=None):

        q = self.to_q(x)     # B*N*(H*C)
        k = self.to_k(key)   # B*M*(H*C)
        v = self.to_v(value) # B*M*(H*C)
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (B*H)*N*M
        self.fill_inf_from_mask(sim, mask)
        attn = sim.softmax(dim=-1) # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)
    
class FrequencyProtos(nn.Module):
    def __init__(self,
                in_chans,
                high_pass=True,
                ):
        super().__init__()
        self.high_pass = high_pass
        self.in_channels = in_chans
    #    self.freq_deform_conv = DeformableConv2d(in_channels=in_channels, out_channels = in_channels)
        self.freq_weight = nn.Conv2d(in_channels = in_chans,
                                             out_channels=1,
                                              stride=1,
                                              kernel_size=3,
                                              padding=1,
                                              bias=True)
      
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.in_channels // 16, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.in_channels // 16, self.in_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
     
        self.act = "sigmoid"
        self.freq_thres = 0.35
        
    def forward(self, x):
        _,_,h,w = x.shape
   #     x = self.freq_deform_conv(x)
        freq_weight = self.freq_weight(x)
        
        if self.act == "sigmoid":
            freq_weight = freq_weight.sigmoid()
            
        elif self.act == "softmax":
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
            
        else:
            raise NotImplementedError
      
        x_fft = torch.fft.fftshift(torch.fft.fft2(x))
       
        if not self.high_pass:
            low_mask = torch.zeros_like(x_fft, device=x_fft.device)
            low_mask[:,:,round(h/2 - h * self.freq_thres):round(h / 2 + h * self.freq_thres), round(w / 2 - w * self.freq_thres):round(w / 2 + w * self.freq_thres)] = 1.0
            low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * low_mask)).real
            low_x_fft = x_fft * low_mask
            low_c_att = torch.sqrt(self.channel_att(low_x_fft.real) ** 2+self.channel_att(low_x_fft.imag) ** 2 + 1e-8)
            low_part = low_part * freq_weight[:,0:1,] * low_c_att
            
            return low_part
            
        elif self.high_pass:
            high_mask = torch.ones_like(x_fft, device=x_fft.device)
            high_mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h*self.freq_thres),round(w/2-w*self.freq_thres):round(w/2 + w * self.freq_thres)] = 0.0
            high_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * high_mask)).real
            high_x_fft = x_fft * high_mask
            high_c_att = torch.sqrt(self.channel_att(high_x_fft.real) ** 2+self.channel_att(high_x_fft.imag) ** 2 + 1e-8)
            high_part = high_part * freq_weight[:, 0:1,] * high_c_att
            return high_part
            
class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    
class BottleneckDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, high=True, resolution=16):
        super(BottleneckDownsample, self).__init__()
        # 第一个 1x1 卷积，减少通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        # 3x3 卷积，提取空间特征，同时下采样
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)
        if high:
            self.freq = FrequencyProtos(in_chans = out_channels // 4, high_pass=True)
        else:
            self.freq = FrequencyProtos(in_chans = out_channels // 4, high_pass=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        
        # 第二个 1x1 卷积，恢复通道数
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # 下采样捷径分支
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.freq(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 捷径分支
        identity = self.downsample(identity)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

class ResConvNets(nn.Module):
    def __init__(self,high=True):
        super(ResConvNets, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.high = high
        # Bottleneck 模块
        self.layer1 = self._make_layer(64, 256, stride=1, resolution=64)
        self.layer2 = self._make_layer(256, 512, stride=2, resolution=64)
        self.layer3 = self._make_layer(512, 1024, stride=2, resolution=32)

    def _make_layer(self, in_channels, out_channels, stride, resolution):
        layers = []
        # 第一个 Bottleneck 模块进行下采样
        layers.append(BottleneckDownsample(in_channels, out_channels, stride,high=self.high, resolution=resolution))
        # 后续 Bottleneck 模块保持分辨率
        layers.append(BottleneckDownsample(out_channels, out_channels, stride=1, high=self.high, resolution= resolution // stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积和下采样
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过 Bottleneck 模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        return x

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, objs):

        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn( self.norm1(x), objs, objs)  
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) ) 
        
        return x 

class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0):
        super().__init__() 
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                input,
                mask = None,):
        h = self.norm1(input)
        h1 = self.self_attn(query=h, key=h, value=h, attn_mask=mask)[0]
        h = h + self.dropout(h1)
        h = self.norm2(h)
        return h
        

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b,  l, -1)

        return self.to_out(out)

class CrossAttentionLayer(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.perceiver_fg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.perceiver_bg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
    def forward(self, x, latents):
        x_fg, x_bg = x
        latents_fg, latents_bg = latents
        out_fg = self.perceiver_fg(x_fg, latents_fg)
        out_bg = self.perceiver_bg(x_bg, latents_bg)
        return out_fg,  out_bg


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)
        
        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class SerialSampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        num_queries=[8, 8, 8],
        embedding_dim=768,
        output_dim=1024,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
    
        self.fg_geometry = convnext_tiny(depths=[3, 3, 9, 3])
   
        
        self.context_bridge = GatedCrossAttentionDense(query_dim=output_dim, key_dim=output_dim, value_dim=output_dim,
                                                       n_heads=output_dim // dim_head, d_head=dim_head)
        self.bg_geometry = convnext_tiny(depths=[3, 3, 9, 3])
        
    def forward(self, x_objs, obboxes, x_bg):
        embed_objs = self.fg_geometry(x_objs)
        embed_context = self.bg_geometry(x_bg)
        
        contrast = embed_context - embed_objs.repeat(x_bg.size(0), 1, 1)
        embed_context = self.context_bridge(embed_context, contrast).mean(0,keepdims=True)
        
        return embed_objs, embed_context
    
def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
