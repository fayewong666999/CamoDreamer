import os, json, random
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from PIL import Image

from pdb import set_trace as ST

import sys

import torch
import torch.nn as nn

sys.path.append("/cpfs/user/wenzhuangwang/CC-Diff-CV/dinov2")
import hubconf

DINOv2_weight_path = "/cpfs/user/wenzhuangwang/FICGen/dinov2/dinov2_vitl14_pretrain.pth"

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

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
        mask = 1. - mask
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



class myCLIPEnc(nn.Module):
    # def __init__(self, model_config='openai/clip-vit-large-patch14', device='cuda'):
    def __init__(self, model_config='/cpfs/user/wenzhuangwang/TIP/ckpt/clip-vit-large-patch14', 
                 device='cuda'):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.processor = AutoProcessor.from_pretrained(model_config)
        self.model = CLIPModel.from_pretrained(model_config).to(device)
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
    

if __name__ == '__main__':
    instance_transforms = transforms.Compose(
        [
            transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    
    mask_transforms = transforms.Compose(
        [
            transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    cache_name = '/cpfs/user/wenzhuangwang/Datasets/camouflage/lakered/train/lakered_image_emb.pt'
    if not os.path.exists(cache_name):
        myDINOv2 = FrozenDinoV2Encoder()
        
        train_root = "/cpfs/user/wenzhuangwang/Datasets/camouflage/lakered/train/images/"
        data = os.listdir(train_root)
        data = [i for i in data if i.endswith(".jpg")]
        
        emb_dict = {}
        sample_data = random.sample(data, 4040)
        for sample in tqdm(sample_data):
            img_name = sample
            emb_dict[img_name] = {}

        #    caption = sample['text']
            img = Image.open(os.path.join(train_root, img_name)).convert('RGB')
            msk =  Image.open(os.path.join(train_root.replace("images", "masks_fg"), img_name[:-3]+"png"))
            
            img = instance_transforms(img).unsqueeze(0)
            msk = mask_transforms(msk).unsqueeze(0)
            
            img_emb = myDINOv2(image=img, mask=msk)
            
            emb_dict[img_name]['img_emb'] = img_emb.detach().cpu()
            # torch.cuda.empty_cache()

        torch.save(emb_dict, cache_name)