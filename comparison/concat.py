import os
from PIL import Image
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

import random
from collections import defaultdict

test = json.load(open("/cpfs/user/wenzhuangwang/Datasets/aerial/dior/annotations/instances_test.json","r"))

images = test["images"]
annos = test["annotations"]
cates = test["categories"]

image2anno = defaultdict(list)
anno2image = {}

for i in images:
    anno2image[i["id"]]=i["file_name"]
    
for anno in annos:
    image2anno[anno2image[anno["image_id"]]].append([anno["category_id"]]+anno["bbox"])
    
catemap = {}
for i in cates:
    catemap[i["id"]] = i["name"]

palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]


#categories = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

#categories = ['holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish']
categories = ["vehicle", "baseballfield", "groundtrackfield", "windmill", "bridge", \
                    "overpass", "ship", "airplane", "tenniscourt", "airport", \
                    "expressway-service-area", "basketballcourt", "stadium", "storagetank", "chimney", \
                    "dam", "expressway-toll-station", "golffield", "trainstation", "harbor"]
#categories = ["motorcycle", "person", "bus", "truck", "car"]
#categories = ["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup","dog","motorbike", "people", "table"]
#categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
#                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

#categories = ["fish","reptiles","arthropoda","corals","mollusk","plants","ruins","garbage","human","robots"]
def obtain_gtimg(name, img=None):
    names = [name]
    for name in names:
        width, height = Image.open(os.path.join("/cpfs/user/wenzhuangwang/Datasets/aerial/dior/test",name)).convert("RGB").size
        image_ = Image.open(os.path.join("/cpfs/user/wenzhuangwang/Datasets/aerial/dior/test", f"{name}")).convert("RGB")
        image_ = image_.resize([512,512])

        labels = image2anno[name]
        if img is not None:
            image_tmp = np.array(img, dtype=np.uint8)
        else:
            image_tmp = np.ones((512, 512, 3), dtype=np.uint8) * 255

        boxes=[]
        for idx,label in enumerate(labels):

            tmp = label[1:]
            x1 = int(tmp[0])
            y1 = int(tmp[1])
            w = int(tmp[2])
            h = int(tmp[3])
            x2 = int(x1+w)
            y2 = int(y1+h)

            x1 = int(x1/width*512)
            y1 = int(y1/height*512)
            x2 = int(x2/width*512)
            y2 = int(y2/height*512)

            xn1 = x1 
            yn1 = y1
            xn2 = x2
            yn2 = y2
            boxes.append([xn1,yn1,xn2,yn2])
        classes = [catemap[i[0]] for i in labels]
     #   classes.append("Add cuttlefish")
     #   classes[0]="->airplane"
     #   classes[2] = "Add storagetank"
     #   classes[4] = "Add storagetank"
        #print(boxes, classes)

        for idx,(box,label) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box
            color = palette[categories.index(label)]
            cv2.rectangle(image_tmp, (x1,y1),(x2,y2), color,2)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,1.0,2)
            cv2.rectangle(image_tmp, (x1,y1-text_height),(x1+text_width,y1),color,-1)
            cv2.putText(image_tmp,label, (x1,y1 - text_height+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2)
        return Image.fromarray(image_tmp)
    





root1 = "/cpfs/user/wenzhuangwang/Datasets/camouflage/lakered/validation/images/"
root2 = "/cpfs/user/wenzhuangwang/CamoRender/comparison/copy/"
#root2 = "/cpfs/shared/L4-new/AIGC/wenzhuangwang/user/LAKERED_Result/LAKERED_Result/images/"
#root3 = "/cpfs/shared/L4-new/AIGC/wenzhuangwang/user/FACIG_images/images/"
#root4 = "/cpfs/user/wenzhuangwang/CT-CIG-main/CTCIG-LAKERED_result/test_output_0729/eval_img/"
root5 = "/cpfs/user/wenzhuangwang/CamoRender/generated_images/lakered_ok_all/"
images = os.listdir(root1)
images = [i for i in images if i.startswith("COD")]

for item in images:
    img1 = Image.open(os.path.join(root1, item)).resize([512,512])
#    img1 = obtain_gtimg(item, img1)
    img2 = Image.open(os.path.join(root2, item)).resize([512,512])
#    img2 = obtain_gtimg(item, img2)
#    img3 = Image.open(os.path.join(root3, item)).resize([512,512])
#    img3 = obtain_gtimg(item, img3)
#    img4 = Image.open(os.path.join(root4, item)).resize([512,512])
#    img4 = obtain_gtimg(item, img4)
    img5 = Image.open(os.path.join(root5, item)).resize([512,512])
#    imagelists = [img1,img2,img3,img4,img5]
    imagelists = [img1, img2, img5]
    new_img = Image.new("RGB",(512*3,512))
    x_offset = 0
    for img in imagelists:
        new_img.paste(img,(x_offset, 0))
        x_offset += 512
    new_img.save(os.path.join("/cpfs/user/wenzhuangwang/CamoRender/comparison/copy/", item))