import functools
import os
import random
import numpy as np
import imagesize
import torch
import torch.nn.functional as F
from PIL import Image
#DIOR
def get_classnames(train_data_dir):
    if "dior" in train_data_dir:
        list_of_name = ["vehicle", "baseballfield", "groundtrackfield", "windmill", "bridge", \
                    "overpass", "ship", "airplane", "tenniscourt", "airport", \
                    "expressway-service-area", "basketballcourt", "stadium", "storagetank", "chimney", \
                    "dam", "expressway-toll-station", "golffield", "trainstation", "harbor"]
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/dior/dior_emb.pt')
        
    elif "exdark" in train_data_dir:
        list_of_name = ["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup","dog","motorbike", "people", "table"]
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/exdark/exdark_emb.pt')
        
    elif "ruod" in train_data_dir:
        list_of_name = ['holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish']
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/ruod/ruod_emb.pt')
    
    elif "dawn" in train_data_dir:
        list_of_name = ["bicycle", "motorcycle", "person", "bus", "truck", "car"]
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/dawn/dawn_emb.pt')
        
    elif "voc" in train_data_dir:
        list_of_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
                        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/voc/voc_emb.pt')
    elif "nuimages" in train_data_dir:
        list_of_name = [
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
                ]
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/nuimages/nuimages_emb.pt')
    elif "acdc" in train_data_dir:
        list_of_name = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/acdc/acdc_emb.pt')
    elif "coco2017" in train_data_dir:
        list_of_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/FICGen/datasets/coco/coco_emb.pt')
    elif "camouflage" in train_data_dir:
        list_of_name = ['foreground']
        data_emb_dict = torch.load('/cpfs/user/wenzhuangwang/Datasets/camouflage/lakered/train/bg_emb.pt')
    return list_of_name, data_emb_dict
"""
for name in list_of_name:
    name_of_dir = os.path.join(ref_img_path, name)
    list_of_image = os.listdir(name_of_dir)
    list_of_image = [i for i in list_of_image if i.endswith("jpg")]
    list_of_image = sorted(list_of_image, 
                           key = lambda img: functools.reduce(lambda x, y: x*y, imagesize.get(os.path.join(name_of_dir, img))), 
                           reverse=True)
    dict_of_images[name] = {img: functools.reduce(lambda x, y: x/y, imagesize.get(os.path.join(name_of_dir, img))) 
                            for img in list_of_image[:200]}
"""

def seed_everything(seed):
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array/value - 1)).argmin()
        return idx

def get_sup_mask(mask_list):
    or_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        or_mask += mask
    or_mask[or_mask >= 1] = 1
    sup_mask = 1 - or_mask
    return sup_mask


data_emb_dict = torch.load("/cpfs/user/wenzhuangwang/Datasets/camouflage/lakered/train/lakered_image_emb.pt")
def get_similar_examplers(query_img_name, prompt_emb, topk=3, sim_mode='img2img'):
    prompt_emb = F.normalize(prompt_emb, dim=-1).detach().cpu()
    # go through the embeddings and get the most similar topk examples
    img_name_list = []
    sim_val_list = []
    
    for img_name, data_emb in data_emb_dict.items():
        img_name_list.append(img_name)
#        txt_emb = data_emb['txt_emb']
        img_emb = data_emb['img_emb']
        if sim_mode == 'text2text':
            sim_val = (prompt_emb * txt_emb).sum(dim=-1)
        elif sim_mode == 'text2img':
            sim_val = (prompt_emb * img_emb).sum(dim=-1)
        elif sim_mode == 'img2img':
            sim_val = (prompt_emb * img_emb).sum(dim=-1)
        elif sim_mode == 'both':
            txt_sim_val = (prompt_emb * txt_emb).sum(dim=-1)
            img_sim_val = (prompt_emb * img_emb).sum(dim=-1)
            sim_val = (txt_sim_val + img_sim_val) * 0.5
        else:
            raise ValueError('Invalid mode for similarity computation! (text2text | text2img | both)') 
        sim_val_list.append(sim_val.item())
    # sort the similarity values and obtain the topk one
    sim_val_list, img_name_list = zip(*sorted(zip(sim_val_list, img_name_list)))
    sim_val_list = list(sim_val_list)
    img_name_list = list(img_name_list)
    img_emb_list = [data_emb_dict[img_name] for img_name in img_name_list[-topk:]]
    return img_name_list[-topk:][::-1]


