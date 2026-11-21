import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
from inspect import isfunction
from einops import rearrange, repeat
from torch import einsum
#from .simple_moe import MoEConvLayer
#from .router_moe import SparseMoE
#from .gcn import GCN, gen_adj_sim

def exists(val):
    return val is not None

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
    def __init__(self, dim, dim_out=None, mult=2, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        self.norm = nn.LayerNorm(dim)
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
        identity = x
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = self.net(x)
        x = x.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return x + identity
    

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor

class Attention_S(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, dropout=0.):
        super().__init__()
        inner_dim = dim
        dim_head = dim // num_heads
        self.heads = num_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
     
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t:rearrange(t, "b n (h d) -> b h n d", h = self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out).transpose(1, 2).reshape(B, C, H, W).contiguous()
        
class Attention_F(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(Attention_F, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())
       
    def forward(self, x):
        b, c, h, w = x.shape
       
        q_f = torch.fft.fft2(x.float())
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())
      

        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = torch.nn.functional.normalize(q_f, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = custom_complex_normalization(attn_f, dim=-1)
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real)*torch.fft.fft2(x.float())))
        out = self.project_out(torch.cat((out_f,out_f_l),1))
        return out
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, freq_attn=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.freq_attn = freq_attn
        if freq_attn:
            self.FreqGate = Attention_F(gate_channels)
            self.project_out = nn.Conv2d(gate_channels * 2, gate_channels, kernel_size=1, bias=True)
            self.act = nn.Sigmoid()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        
        if self.freq_attn:
            x_freq = self.FreqGate(x)
            gated = self.act(self.project_out(torch.cat([x_freq, x_out], dim=1)))
            return gated * x_freq + (1 - gated) * x_out
        else:
            return x_out
        
class GenAdj(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.guidance_transform = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.matrix_transform = nn.Linear(dim * 2, dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        f = x.mean(1, keepdim=True)
        guide_f = self.guidance_transform(f)
        guide_f = self.relu(guide_f)
        guide_f = guide_f.expand(guide_f.size(0), x.size(1), x.size(2))
        x = torch.cat([guide_f, x], dim=-1)
        trans_f = self.matrix_transform(x)
        return trans_f.transpose(1, 2).view(B, C, H, W).contiguous()
        

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gated = nn.Conv2d(dim * 2, dim, 3, 1, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x1, x2):
        gate = self.gated(torch.cat([x1, x2], dim=1))
        gate = self.act(gate)
        return x1 * gate + x2 * (1 - gate)

class ASFA(nn.Module):
    def __init__(self, C, number_pro=30):
        super().__init__()
        self.C = C
        self.number_pro = number_pro
        self.conv1 = nn.Conv2d(C + 1, C, 1, 1)
#        self.gcn = GCN(C,C,C)
        self.cbam1 = CBAM(C, freq_attn=False)
        self.sattn = Attention_S(C)
        self.fattn = Attention_F(C)
        self.gate = nn.Sequential(
            nn.Conv2d(2*C, 2*C, 3, 1, 1, groups=2*C),
            nn.Conv2d(2*C, C, 1)
        )
        self.norm = nn.LayerNorm(C, elementwise_affine=False, eps=1e-6)
 #       self.moe1 = SparseMoE(C)
 #       self.conv2 = nn.Conv2d(C, 1, 1, 1)
 #       self.cbam2 = CBAM(number_pro, reduction_ratio=1)
    
            
    def forward(self, x, guidance_mask, sac_scale=None):
        '''
        :param x: (B, phase_num, HW, C)
        :param guidance_mask: (B, phase_num, H, W)
        :return:
        '''
        
        B, phase_num, HW, C = x.shape
        _, _, H, W = guidance_mask.shape
        guidance_mask = guidance_mask.view(guidance_mask.shape[0], phase_num, -1)[
            ..., None]  # (B, phase_num, HW, 1)

        null_x = torch.zeros_like(x[:, [0], ...]).to(x.device)
        null_mask = torch.zeros_like(guidance_mask[:, [0], ...]).to(guidance_mask.device)

        x = torch.cat([x, null_x], dim=1)
        guidance_mask = torch.cat([guidance_mask, null_mask], dim=1)
        phase_num += 1


        scale = torch.cat([x, guidance_mask], dim=-1)  # (B, phase_num, HW, C+1)
        scale = scale.view(-1, H, W, C + 1)  # (B * phase_num, H, W, C+1)
        scale = scale.permute(0, 3, 1, 2).contiguous()  # (B * phase_num, C+1, H, W)
        
        
        scale = self.conv1(scale)  # (B * phase_num, C, H, W)
        scale_cbam = self.cbam1(scale)
        scale = scale.flatten(2).transpose(1, 2).contiguous()
        scale = self.norm(scale)
        scale = scale.transpose(1, 2).reshape(-1, C, H, W).contiguous()
        scale_attn = self.gate(torch.cat([self.sattn(scale), self.fattn(scale)], dim=1))    # (B * phase_num, C, H, W)
        scale = scale_cbam + scale_attn
    
        scale = scale.view(B, phase_num, C, H * W).transpose(-1, -2).contiguous()  # (B, phase_num, H, W)

        null_scale = scale[:, [-1], ...]
        scale = scale[:, :-1, ...]
        x = x[:, :-1, ...]
        scale = scale.softmax(dim=1)  # (B, phase_num, HW, 1)
        out = (x * scale).sum(dim=1)  # (B, 1, HW, C)
        return out, scale


    
class SAC_plus(nn.Module):
    def __init__(self, C, number_pro=30):
        super().__init__()
        self.C=C
        self.number_pro = number_pro
        self.conv2 = nn.Conv2d(C,1,1,1)
        self.cbam2 = CBAM(number_pro, reduction_ratio=1)
    
    def forward(self, x, guidance_mask=None, sac_scale=None):
        B, phase_num, HW, C = x.shape
        H = W = int(math.sqrt(HW))
        null_x = torch.zeros_like(x[:, [0], ...]).to(x.device)
        x = torch.cat([x, null_x], dim=1)
        phase_num += 1
        scale = x
        scale = scale.view(-1, H, W, C)
        scale = scale.permute(0, 3, 1, 2)
        scale = self.conv2(scale)
        scale = scale.view(B, phase_num, H, W)
        
        null_scale = scale[:, [-1], ...]
        scale = scale[:, :-1, ...]
        x = x[:, :-1, ...]
        
        pad_num = self.number_pro - phase_num + 1
        ori_phase_num = scale[:, :-1, ...].shape[1]
        phase_scale = torch.cat([scale[:,:-1,...],null_scale.repeat(1, pad_num, 1, 1)], dim=1)
        shuffled_order = torch.randperm(phase_scale.shape[1])
        inv_shuffled_order = torch.argsort(shuffled_order)
        
        random_phase_scale = phase_scale[:, shuffled_order, ...]
        scale = torch.cat([random_phase_scale, scale[:,[-1], ...]], dim=1)
        
        scale = self.cbam2(scale)
        scale = scale.view(B, self.number_pro, HW)[...,None]
        random_phase_scale = scale[:, :-1, ...]
        phase_scale = random_phase_scale[:, inv_shuffled_order[:ori_phase_num],:]
        if sac_scale is not None:
            instance_num = len(sac_scale)
            for i in range(instance_num):
                phase_scale[:, i, ...] = phase_scale[:, i, ...] * sac_scale[i]
                
        scale = torch.cat([phase_scale, scale[:, [-1], ...]], dim=1)
        scale = scale.softmax(dim=1)
        out = (x * scale).sum(dim=1)
        return out, scale
    
    
class _FrequencyMix(nn.Module):
    def __init__(self,
                in_channels,
                k_list=[2],
                fs_feat="feat",
                lp_type="freq_channel_att",
                act="sigmoid",
                channel_res=True,
                spatial="conv",
                spatial_group=1,
                compress_ratio=16,
                ):
        super().__init__()
        k_list.sort()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        self.channel_res = channel_res
        if spatial_group > 64: spatial_group = in_channels
        self.spatial_group = spatial_group
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // compress_ratio, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // compress_ratio, 2, 1)
        )
        self.freq_weight_conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = (len(k_list) + 1) * self.spatial_group,
                stride=1,
                kernel_size=3, padding=1, bias=True
            )
        
        self.channel_att_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.in_channels // compress_ratio, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.in_channels // compress_ratio, self.in_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.channel_att_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.in_channels // compress_ratio, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // compress_ratio, self.in_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()     
        )
        self.act = act
    
    def forward(self, x):
        
        B, phase_num, N, C = x.shape
        H =  W = int(math.sqrt(N))
        x_in = rearrange(x, "b num (h w) c->(b num) c h w", h=H)
        x_in = self.stem(x_in)
        
        freq_weight = self.freq_weight_conv(x_in)
        x_gap = F.adaptive_avg_pool2d(x_in, 1)
        rh_rw = self.gate(x_gap)
        rh_rw = torch.sigmoid(rh_rw)
        rh = rh_rw[:, 0:1]
        rw = rh_rw[:, 1:2]

        # ===== 激活 =====
        if self.act == "sigmoid":
            freq_weight = freq_weight.sigmoid()
        elif self.act == "softmax":
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError

        # ===== FFT =====
        x_fft = torch.fft.fftshift(torch.fft.fft2(x_in))

        # ===== 生成坐标 =====
        Bx = x_in.shape[0]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        yy = yy.unsqueeze(0).unsqueeze(0).expand(Bx, 1, H, W)
        xx = xx.unsqueeze(0).unsqueeze(0).expand(Bx, 1, H, W)

        # ===== window大小（连续，不用long）=====
        h_win = rh * 0.5   # normalized
        w_win = rw * 0.5

        # ===== ⭐ soft-rectangle mask（关键）=====
        alpha = 12.0  # 控制接近硬mask程度（10~20都行）

        mask_y = torch.sigmoid(alpha * (h_win - torch.abs(yy)))
        mask_x = torch.sigmoid(alpha * (w_win - torch.abs(xx)))

        low_mask = mask_y * mask_x
        high_mask = 1.0 - low_mask
        
    #    _, _, h, w = x.shape
    #    low_mask[:,:,round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), round(w/2 - w * self.freq_thres): round(w / 2 + w * self.freq_thres)] = 1.0
    #    high_mask[:,:,round(h / 2 - h * self.freq_thres):round(h / 2 + h * self.freq_thres), round(w / 2 - w * self.freq_thres):round(w / 2 + w * self.freq_thres)] = 0.0
        low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft*low_mask)).real
        high_part = x_in - low_part
        low_x_fft = x_fft * low_mask
        high_x_fft = x_fft * high_mask
        low_c_att = torch.sqrt(self.channel_att_low(low_x_fft.real) ** 2 + self.channel_att_low(low_x_fft.imag)**2+1e-8)
        high_c_att = torch.sqrt(self.channel_att_high(high_x_fft.real) ** 2 + self.channel_att_high(high_x_fft.imag)**2+1e-8)
        low_part = low_part * freq_weight[:, 0:1,] * low_c_att
        high_part = high_part * freq_weight[:, 1:2,] * high_c_att
        res = low_part+high_part
        res = res + x_in
        res = res.view(B, phase_num, C, H * W).transpose(-1, -2).contiguous()
        res = res.softmax(dim=1)  # (B, phase_num, HW, 1)
        
        res = (x * res).sum(dim=1)  # (B, 1, HW, C)
        
        return res
    
    
    