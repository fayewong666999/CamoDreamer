import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math
import numpy as np
from PIL import Image
from inspect import isfunction
from einops import rearrange, repeat
from torch import einsum
from .aggrefusion import ASFA, SAC_plus, _FrequencyMix
import matplotlib.pyplot as plt

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, return_attn=False, need_softmax=True, visz=False):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if need_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_attn:
            attn = attn.view(b, h, attn.shape[-2], attn.shape[-1])
            return self.to_out(out), attn
        else:
            return self.to_out(out)

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)  # torch.Size([5, 30, 64])

class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 8  # 2 is sin&cos, 8 is xyxyxyxy

        # -------------------------------------------------------------- #
        self.linears_position = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, boxes):

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*1*4 --> B*1*C torch.Size([5, 1, 64])
        xyxy_embedding = self.linears_position(xyxy_embedding)  # B*1*C --> B*1*768 torch.Size([5, 1, 768])

        return xyxy_embedding

class LayoutAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., use_lora=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.use_lora = use_lora
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, return_attn=False, need_softmax=True, guidance_mask=None):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        _, phase_num, H, W = guidance_mask.shape
        HW = H * W
        guidance_mask_o = guidance_mask.view(b * phase_num, HW, 1)
        guidance_mask_t = guidance_mask.view(b * phase_num, 1, HW)
        guidance_mask_sim = torch.bmm(guidance_mask_o, guidance_mask_t)  # (B * phase_num, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, phase_num, HW, HW).sum(dim=1)
      
        guidance_mask_sim[guidance_mask_sim > 1] = 1  # (B, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, 1, HW, HW)
        guidance_mask_sim = guidance_mask_sim.repeat(1, self.heads, 1, 1)
        guidance_mask_sim = guidance_mask_sim.view(b * self.heads, HW, HW)  # (B * head, HW, HW)

        sim[:, :, :HW][guidance_mask_sim == 0] = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of

        if need_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim
            
            
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_attn:
            attn = attn.view(b, h, attn.shape[-2], attn.shape[-1])
            return self.to_out(out), attn
        else:
            return self.to_out(out)

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

class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)
    
class RefinedShader(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, ca_x, guidance_mask, other_info,  return_fuser_info=False, ca_scale=None):
  
        full_H = other_info["height"]
        full_W = other_info["width"]
        B,_,HW,C = ca_x.shape
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode="bilinear")
        guidance_mask = torch.cat([torch.ones(B, 1, H, W).to(guidance_mask.device), guidance_mask * 10], dim=1)
        guidance_mask = guidance_mask.view(B, instance_num + 1, HW, 1)
        out_MIGC = (ca_x * guidance_mask).sum(dim=1) / (guidance_mask.sum(dim=1) + 1e-6)
        if return_fuser_info:
            return out_MIGC, None
        else:
            return out_MIGC
        
class MIGC_plus(nn.Module):
    def __init__(self, C, attn_type="base", context_dim=768, heads=8):
        super().__init__()
        self.ea = CrossAttention(query_dim=C, context_dim=context_dim,
                                heads=heads, dim_head=C//heads,
                                dropout=0.0)
        self.la = LayoutAttention(query_dim=C,
                                 heads=heads, dim_head=C//heads,
                                 dropout=0.0)
        self.norm = nn.LayerNorm(C)
        self.sac = SAC_plus(C)
        self.pos_net = PositionNet(in_dim=768,out_dim=768)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.refined_shader=RefinedShader()
    
    def forward(self, ca_x, guidance_mask, other_info, return_fuser_info=False):
        ca_scale = other_info["ca_scale"] if "ca_scale" in other_info else None
        ea_scale = other_info["ea_scale"] if "ea_scale" in other_info else None
        mig_ca_x = self.refined_shader(ca_x, guidance_mask, other_info, ca_scale=ca_scale)
        full_H = other_info["height"]
        full_W = other_info["width"]
        B, _, HW, C = ca_x.shape
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = F.interpolate(guidance_mask, size=(H,W), mode="bilinear")
        supplement_mask = other_info["supplement_mask"]
        supplement_mask = F.interpolate(supplement_mask, size=(H,W), mode="bilinear")
        image_token = other_info["image_token"]
        assert image_token.shape == ca_x.shape
        context = other_info["context_pooler"]
        box = other_info["box"]
        box = box.reshape(B * instance_num, 1, -1)
        box_token = self.pos_net(box)
       
        context = torch.cat([context[1:, ...], box_token], dim=1)
        sac_scale = other_info["sac_scale"] if "sac_scale" in other_info else None
        
        ea_x, ea_attn = self.ea(self.norm(image_token[:, 1:, ...].reshape(B * instance_num, HW, C)), context=context, return_attn=True)
        ea_x = ea_x.view(B, instance_num, HW, C)
        ea_x = ea_x * guidance_mask.view(B, instance_num, HW, 1)
        if ea_scale is not None:
            assert len(ea_scale) == instance_num
            for i in range(instance_num):
                ea_x[:, i, ...] = ea_x[:, i, ...] * ea_scale[i]
        
        ori_image_token = image_token[:,0,...]
        fusion_template = self.la(x=ori_image_token, guidance_mask=torch.cat([guidance_mask[:,:, ...], supplement_mask], dim=1))
        fusion_template = fusion_template.view(B, 1, HW, C)
        shading_instances_and_template = torch.cat([ea_x, fusion_template], dim=1)
        out_MIGC, sac_scale = self.sac(shading_instances_and_template, sac_scale=sac_scale)
        out_MIGC = mig_ca_x + out_MIGC * torch.tanh(self.gamma) * other_info.get("gamma_scale", 1.0)
        if return_fuser_info:
            fuser_info = {}
            fuser_info["sac_scale"] = sac_scale.view(B, instance_num + 1, H, W)
            fuser_info["ea_attn"] = ea_attn.mean(dim=1).view(B, instance_num, H, W, 2)
            return out_MIGC, fuser_info
        else:
            return out_MIGC


class VFEAttention(nn.Module):
    def __init__(self, C, attn_type='base', context_dim=768, heads=8):
        # context_dim: SD1.4 768  SD2.1 1024
        super().__init__()
        self.ea_obj = CrossAttention(query_dim=C, context_dim=context_dim,
                                 heads=heads, dim_head=C // heads,
                                 dropout=0.0)
        self.norm_obj = nn.LayerNorm(C)
        self.ea_bg = CrossAttention(query_dim=C, context_dim=context_dim,
                                 heads=heads, dim_head=C // heads,
                                 dropout=0.0)
        self.norm_bg = nn.LayerNorm(C)

        self.la = LayoutAttention(query_dim=C, heads=heads, 
                                  dim_head=C // heads, dropout=0.0)
        self.asfa = _FrequencyMix(C)
        self.linear = nn.Linear(C, context_dim)
        self.fg_pool = nn.Sequential(
            nn.Linear(context_dim, context_dim // 16),
            nn.GELU(),
            nn.Linear(context_dim // 16, 1),
            nn.Softmax(dim=1),
        )

    
    def forward(self, ca_x, guidance_mask, other_info):
        # x: (B, instance_num+1, HW, C)
        # guidance_mask: (B, instance_num, H, W)
        # box: (instance_num, 4)
        # image_token: (B, instance_num+1, HW, C)
        full_H = other_info['height']
        full_W = other_info['width']
        B, _, HW, C = ca_x.shape
        instance_num = guidance_mask.shape[1]
      
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode='bilinear')   # (B, instance_num, H, W)
        
   #     guidance_mask = torch.nn.AvgPool2d(down_scale // 8,down_scale // 8)(guidance_mask)
        
        supplement_mask = other_info['supplement_mask']  # (B, 1, 64, 64)
        supplement_mask = F.interpolate(supplement_mask, size=(H, W), mode='bilinear')  # (B, 1, H, W)
    #    supplement_mask = torch.nn.AvgPool2d(down_scale // 8,down_scale // 8)(supplement_mask)
        image_token = other_info['image_token']
        assert image_token.shape == ca_x.shape
        context = other_info['context_pooler']
        box = other_info['box']
        box = box.view(B * instance_num, 1, -1)
        
        # add reference image feature as condition
        img_features, bg_features = other_info['ref_features']
        
        context_fg = torch.cat([img_features], dim=1)
     
        ea_x, _ = self.ea_obj(self.norm_obj(image_token[:, 1:, ...].view(B * instance_num, HW, C)),
                                context=context_fg, return_attn=True, visz=False)
        ea_x = ea_x.view(B, instance_num, HW, C)
        instance_maps = other_info['instance_maps']
        instance_maps = F.interpolate(instance_maps, size=(H, W), mode='bilinear')
 
        ea_x = ea_x * instance_maps.view(B, instance_num, HW, 1)
         
        ca_x[:, 1:, ...] = ea_x
        
        
   
        context_bg = torch.cat([bg_features], dim=1)
        ###
        fg_cond = self.linear(ea_x[0])
        w = self.fg_pool(fg_cond)
        fg_pool = (w * fg_cond).sum(dim=1, keepdims=True)
        context_bg = context_bg + fg_pool
        ###
        ea_x_bg, _ = self.ea_bg(self.norm_bg(image_token[:, 0, ...].view(B * 1, HW, C)),
                             context=context_bg, return_attn=True)
        
        ea_x_bg = ea_x_bg * supplement_mask.view(B, HW, 1)
        
    
        ca_x[:, 0, ...] = ea_x_bg
        
  
        fusion_template = self.la(x=image_token[:, 0, ...], guidance_mask=torch.cat([guidance_mask[:, :, ...], supplement_mask], dim=1))  # (B, HW, C)
        fusion_template = fusion_template.view(B, 1, HW, C)  # (B, 1, HW, C)
        ca_x = torch.cat([ca_x, fusion_template], dim = 1)
        out_MIGC = self.asfa(ca_x)
        return out_MIGC

    