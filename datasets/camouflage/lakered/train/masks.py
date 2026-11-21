import os
from PIL import Image
import numpy as np

images = os.listdir("images")
images = [i for i in images if i.endswith(".jpg")]

for i in images:
    img = np.array(Image.open(os.path.join("images",i)).convert("RGB"))
    msk = np.array(Image.open(os.path.join("masks_fg",i.split(".")[0]+".png")))
    ys, xs = np.where(msk==255)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    img_crop = img[y_min:y_max+1,x_min:x_max+1]
    mask_crop = msk[y_min:y_max+1,x_min:x_max+1]
    img_crop = np.array(Image.fromarray(np.uint8(img_crop)).resize([512,512],Image.BILINEAR))
    mask_crop = np.array(Image.fromarray(np.uint8(mask_crop)).resize([512,512],Image.NEAREST))
    fg_pixels = img_crop[mask_crop==255]
    mean_color = fg_pixels.mean(axis=0).astype(np.uint8)
    output = img_crop.copy()
    output[mask_crop==0] = 255

    Image.fromarray(np.uint8(output)).save(os.path.join("mix_fg_ori", i))
'''

import os
from PIL import Image
import numpy as np

images = os.listdir("images")
images = [i for i in images if i.endswith(".jpg")]

for i in images:
    img = np.array(Image.open(os.path.join("images",i)).convert("RGB"))
    msk = np.array(Image.open(os.path.join("masks_fg",i.split(".")[0]+".png")).convert("L"))
    fg_mask = msk==255
    fg_pixels = img[fg_mask]
    mean_color = fg_pixels.mean(axis=0)
    image = img.copy()
    image[~fg_mask] = mean_color

    Image.fromarray(np.uint8(image)).save(os.path.join("mix_fg_ori", i))
    
'''