import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPModel, AutoTokenizer, AutoProcessor, CLIPImageProcessor
from PIL import Image


sys.path.append("/cpfs/user/wenzhuangwang/FICGen/dinov2")
import hubconf

from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, AutoProcessor

DINOv2_weight_path = "/cpfs/user/wenzhuangwang/FICGen/dinov2/dinov2_vitl14_pretrain.pth"

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__() 

    def encode(self, *args, **kwargs):
        raise NotImplementedError
        

class myCLIPEnc(nn.Module):
    # def __init__(self, model_config='openai/clip-vit-large-patch14', device='cuda'):
    def __init__(self, model_config='/cpfs/user/wenzhuangwang/TIP/ckpt/clip-vit-large-patch14', 
                 device='cuda'):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.processor = AutoProcessor.from_pretrained(model_config)
        self.model = CLIPModel.from_pretrained(model_config, use_safetensors=False).to(device)
        self.model.eval()

    def forward(self, caption=None, img=None):
        if caption is not None:
            txt_inp  = self.tokenizer(caption, padding=True, truncation=True, return_tensors="pt").to(self.device) # pad and truncate to the max_length
            txt_feat = self.model.get_text_features(**txt_inp)
            txt_feat = F.normalize(txt_feat, dim=-1).detach().cpu()
        else:
            txt_feat = None

        if img is not None:
            img_inp  = self.processor(images=img, return_tensors="pt").to(self.device)
            img_feat = self.model.get_image_features(**img_inp)
            img_feat = F.normalize(img_feat, dim=-1).detach().cpu()
        else:
            img_feat = None
        
        return txt_feat, img_feat
    
class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, device="cuda", freeze=True):
        super().__init__()
        dinov2 = hubconf.dinov2_vitl14() 
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=True)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        # self.projector = nn.Linear(1536, 768)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, mask):
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        mask = mask.to(self.device)
        features = self.model.forward_features(image)
        
        tokens = features["x_norm_patchtokens"]
        image_features  = features["x_norm_clstoken"]
        mask = F.interpolate(mask, size=(16, 16), mode="nearest")
        mask = mask.flatten(2).transpose(1, 2).contiguous()
        image_features = (tokens * mask).mean(1)
        hint = F.normalize(image_features, dim=-1).detach().cpu()
     #   hint = torch.cat([image_features,tokens],1) # 8,257,1024
        # hint = self.projector(hint)
        return hint

    def encode(self, image):
        return self(image)

    
    
class FrozenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the CLIP encoder for image
    """
    def __init__(self, device="cuda", freeze=True, model_name = "/cpfs/user/wenzhuangwang/TIP/ckpt/clip-vit-large-patch14/"):
        super().__init__()
        
        clipvision = CLIPVisionModel.from_pretrained(model_name, use_safetensors=False)
        self.selected_layers = [24]
     #   self.selected_layers = [16, 18, 20, 24]
        self.model = clipvision.to(device)
        self.processor = CLIPImageProcessor()
        self.device = device
        if freeze:
            self.freeze()
       
    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img, clstoken=True, reward=False):
       
        if isinstance(img,list):
            img = torch.cat(img,0)
            
        img = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
    
        outputs = self.model(img)
        hidden_states = outputs.last_hidden_state
        if not clstoken:
            image_features = hidden_states[:, 1:, :]
            B,N,D = image_features.shape
            H =  W = int(N**0.5)
            hint = image_features.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        else:
            hint = hidden_states[:, :, :]
        
        # hint = self.projector(hint)
        if not reward:
            return hint
        else:
            return hint[:,1:].reshape(b, h//14, w//14, -1).permute(0,3,1,2)

    def encode(self, image):
        return self(image)
    
class CLIPTextEncoder(AbstractEncoder):
    """
    Uses the CLIP encoder for text
    """
    def __init__(self, device="cuda", freeze=True, model_name = "/cpfs/user/wenzhuangwang/aerogen/ckpt/clip/clip-vit-large-patch14/", max_length=77, selected_layers=[6, 8, 10, 12]):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.transformer = CLIPTextModel.from_pretrained(model_name, output_hidden_states=True, use_safetensors=False).to(device)
        self.device = device
        self.max_length = max_length
        self.selected_layers = selected_layers
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, texts):
        texts = texts[0][1:]
        batch_encoding = self.tokenizer(texts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        attention_mask = batch_encoding["attention_mask"].to(self.device)
        outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        multi_level_embeddings = [hidden_states[i] for i in self.selected_layers]
        z = torch.cat(multi_level_embeddings, dim=-1)
        
        return z
            
    def encode(self, text):
        return self.forward(text)

    

    
    
def compute_mean_cov(feats:torch.Tensor, eps:float=1e-6):
    mu = feats.mean(dim=0)
    xm = feats - mu
    N = feats.shape[0]
    if N > 1:
        cov = (xm.T@xm) / (N - 1)
    else:
        cov = torch.zeros((feats.size(1), feats.size(1)), device=feats.device, dtype=feats.dtype)
    cov += eps * torch.eye(feats.size(1), device=feats.device, dtype=feats.dtype)
    return mu, cov


def matrix_sqrt(mat:torch.Tensor, eps:float=1e-6):
    mat = (mat + mat.T) / 2
    device = mat.device
    vals, vecs = torch.linalg.eigh(mat.cpu())
    vals, vecs = vals.to(device), vecs.to(device)
 #   U, S, Vh = torch.linalg.svd(mat)
 #   S = torch.clamp(S, min=0.0)
 #   sqrt_S = torch.sqrt(S + eps)
    vals = torch.clamp(vals, min=0.0)
    sqrt_vals = torch.sqrt(vals + eps)
    return (vecs * sqrt_vals.unsqueeze(0)) @ vecs.T

def gaussian_wasserstein_distance(mu_r, cov_r, mu_s, cov_s):
    mean_term = torch.sum((mu_r - mu_s) ** 2)
    covmean = matrix_sqrt(cov_r@cov_s)
    trace_term = torch.trace(cov_r + cov_s - 2 * covmean)
    trace_term = torch.clamp(trace_term, min=0.0)
    return mean_term + trace_term


def gwd_clip_loss(real_feats:torch.Tensor, synth_feats:torch.Tensor, eps:float=1e-6,scale=600.0):
    real_feats = real_feats.flatten(2).transpose(1, 2).contiguous()
    synth_feats = synth_feats.flatten(2).transpose(1, 2).contiguous()

    real_feats = real_feats.reshape(-1, real_feats.size(-1)).contiguous()
    synth_feats = synth_feats.reshape(-1, synth_feats.size(-1)).contiguous()
 
    mu_r, cov_r = compute_mean_cov(real_feats, eps)
    mu_s, cov_s = compute_mean_cov(synth_feats, eps)
    gwd = gaussian_wasserstein_distance(mu_r, cov_r, mu_s, cov_s)
    if scale is None:
        scale = gwd.detach() + eps

    loss = 1.0 - torch.exp(-gwd / scale)
    return loss
    
        
        
    