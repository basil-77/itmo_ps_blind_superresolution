import os

import numpy as np
import random

import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class image_dataset_from_file(Dataset):
    def __init__(self, source_path, ref_path, size=None, ref_scale=2, source_mask=False, ref_mask=False):
        self.source_path = source_path
        self.ref_path = ref_path
        self.size = size
        self.ref_scale = ref_scale
        self.source_mask = source_mask
        self.ref_mask = ref_mask
        self.images_source = []
        self.images_ref = []
        self.images_source_img = []
        self.images_ref_img = []
                
        for filename in os.listdir(self.source_path):
            if self.source_mask:
                if filename.find(self.source_mask)>0:
                    self.images_source.append(os.path.join(self.source_path, filename))
            else:
                self.images_source.append(os.path.join(self.source_path, filename))


        for filename in os.listdir(self.ref_path):
            if self.ref_mask:
                if filename.find(self.ref_mask)>0:
                    self.images_ref.append(os.path.join(self.ref_path, filename))
            else:
                self.images_ref.append(os.path.join(self.ref_path, filename))

        
        image_transforms = []
        image_transforms.append(torchvision.transforms.v2.ToImage())
        image_transforms.append(torchvision.transforms.v2.ToDtype(torch.float32, scale=True))
        if self.size:
            image_transforms.append(torchvision.transforms.Resize(size,
                                                                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                  antialias=True))
        self.transform = torchvision.transforms.Compose(image_transforms)
        
        for image in tqdm(self.images_source):
            img = read_image(image)
            img_tr = self.transform(img)
            self.images_source_img.append(img_tr)
            
        for image in tqdm(self.images_ref):
            img = read_image(image)
            img_tr = self.transform(img)
            self.images_ref_img.append(img_tr)

        
    def __len__(self):
        return len(self.images_source)
    
    def __getitem__(self, index):
        img = read_image(self.images_source[index])
        img = self.transform(img)
        data_source = img
        img = read_image(self.images_ref[index])
        img = self.transform(img)
        data_ref = img
        return data_source, data_ref
        

class image_dataset_patches_from_file(Dataset):
    def __init__(self, source_path, target_path, ref_path, patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        self.source_path = source_path if type(source_path)==list else [source_path]*1
        self.target_path = target_path if type(target_path)==list else [target_path]*1
        self.ref_path = ref_path if type(ref_path)==list else [ref_path]*1
        self.patch_size = patch_size
        self.ref_scale = ref_scale
        self.normalize = normalize
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.ref_mask = ref_mask
        self.refastarget = refastarget
        self.images_source = []
        self.images_target = []
        self.images_ref = []
        self.images_source_img = []
        self.images_target_img = []
        self.images_ref_img = []
        self.images_source_patches = []
        self.images_target_patches = []
        self.images_ref_patches = []
        
        
        for path in self.source_path:        
            for filename in os.listdir(path):
                if self.source_mask:
                    if filename.find(self.source_mask)>0:
                        self.images_source.append(os.path.join(path, filename))
                else:
                    self.images_source.append(os.path.join(path, filename))

        for path in self.target_path:
            for filename in os.listdir(path):
                if self.target_mask:
                    if filename.find(self.target_mask)>0:
                        self.images_target.append(os.path.join(path, filename))
                else:
                    self.images_target.append(os.path.join(path, filename))

        for path in self.ref_path:
            for filename in os.listdir(path):
                if self.ref_mask:
                    if filename.find(self.ref_mask)>0:
                        self.images_ref.append(os.path.join(path, filename))
                else:
                    self.images_ref.append(os.path.join(path, filename))
                
        if limit:
            self.images_source = self.images_source[:limit]
            self.images_target = self.images_target[:limit]
            self.images_ref = self.images_ref[:limit]
            
            
            
        image_transforms = []
        image_transforms.append(torchvision.transforms.v2.ToImage())
        image_transforms.append(torchvision.transforms.v2.ToDtype(torch.float32, scale=True))
        self.transform = torchvision.transforms.Compose(image_transforms)
        
        for image in tqdm(self.images_source):
            img = read_image(image)[:3]
            if img.shape[0]!= 3:
                img = img.expand(3,-1,-1)
            if (img.shape[1]>=patch_size[0] and img.shape[2]>=patch_size[1]):
                img_tr = self.transform(img)
                self.images_source_img.append(img_tr)
                patches = img_tr.data.unfold(0, 3, 3).unfold(1, patch_size[0], patch_size[1]).unfold(2, patch_size[0], patch_size[1])
                size = patches[0].shape
                for i in range(size[0]):
                    for j in range(size[1]):
                        self.images_source_patches.append(patches[0][i][j])
            else:
                self.images_source.remove(image)
                
                        
        for image in tqdm(self.images_ref):
            img = read_image(image)[:3]
            if img.shape[0]!= 3:
                img = img.expand(3,-1,-1)
            if (img.shape[1]>=patch_size[0]*self.ref_scale and img.shape[2]>=patch_size[1]*self.ref_scale):    
                img_tr = self.transform(img)
                self.images_ref_img.append(img_tr)
                patches = img_tr.data.unfold(0, 3, 3).unfold(1,
                                                             patch_size[0]*self.ref_scale,
                                                             patch_size[1]*self.ref_scale).unfold(2,
                                                                                                  patch_size[0]*self.ref_scale,
                                                                                                  patch_size[1]*self.ref_scale)
                size = patches[0].shape
                for i in range(size[0]):
                    for j in range(size[1]):
                        self.images_ref_patches.append(patches[0][i][j])
            else:
                self.images_ref.remove(image)

        if self.refastarget:
            self.target_path = self.ref_path
            self.target_mask = self.ref_mask
            self.images_target = self.images_ref
            self.images_target_img = self.images_ref_img
            self.images_target_patches = self.images_ref_patches
        else:
            for image in tqdm(self.images_target):
                img = read_image(image)[:3]
                if img.shape[0]!= 3:
                    img = img.expand(3,-1,-1)
                if (img.shape[1]>=patch_size[0] and img.shape[2]>=patch_size[1]):
                    img_tr = self.transform(img)
                    self.images_target_img.append(img_tr)
                    patches = img_tr.data.unfold(0, 3, 3).unfold(1, patch_size[0], patch_size[1]).unfold(2, patch_size[0], patch_size[1])
                    size = patches[0].shape
                    for i in range(size[0]):
                        for j in range(size[1]):
                            self.images_target_patches.append(patches[0][i][j])
                else:
                    self.images_target.remove(image)
                  

                
    def __len__(self):
        return len(self.images_source_patches)
    
    def __getitem__(self, index):
        data_source = self.images_source_patches[index]
        data_target = self.images_target_patches[index]
        return data_source, data_target


class data_images_lrsonshrfather(image_dataset_from_file):
    def __init__(self,
                                 lr_path,
                                 hr_path,
                                 lr_mask=None,
                                 hr_mask=None,
                                 scale=2,
                                 scales=[2,3,4],
                                 ):
        super(data_images_lrsonshrfather, self).__init__(source_path=lr_path, ref_path=hr_path, size=None, ref_scale=scale, source_mask=lr_mask, ref_mask=hr_mask) 
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.scale = scale
        self.scales = scales
        self.images_lr_sons = []
        self.images_hr_fathers = []
        
        scale_son = self.scale

        for image in tqdm(self.images_source_img):
            for scale in self.scales:    
                resizes = {
                    'bilinear': torchvision.transforms.Resize((int(image.shape[1]/scale), int(image.shape[2]/scale)),
                                                             interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True),
                    'nearest': torchvision.transforms.Resize((int(image.shape[1]/scale), int(image.shape[2]/scale)),
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                             antialias=True),
                    'nearest_exact': torchvision.transforms.Resize((int(image.shape[1]/scale), int(image.shape[2]/scale)),
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
                                                             antialias=True),
                    'bicubic': torchvision.transforms.Resize((int(image.shape[1]/scale), int(image.shape[2]/scale)),
                                                             interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                             antialias=True),  
                }
                resize_as = random.choice(list(resizes.values()))
                image_downscale = resize_as(image)
                if image_downscale.shape[1] % 2 != 0 or image_downscale.shape[2] %2 != 0:
                    new_h = image_downscale.shape[1] - 1 if image_downscale.shape[1] % 2 != 0 else image_downscale.shape[1]
                    new_w = image_downscale.shape[2] - 1 if image_downscale.shape[2] % 2 != 0 else image_downscale.shape[2]
                    crop = torchvision.transforms.CenterCrop((new_h, new_w))
                    image_downscale = crop(image_downscale)
                    
                augmenters = {
                    #'Rotate90': torchvision.transforms.RandomRotation(degrees=(90,90), expand=True),
                    #'Rotate180': torchvision.transforms.RandomRotation(degrees=(180,180), expand=True),
                    #'Rotate270': torchvision.transforms.RandomRotation(degrees=(270,270), expand=True),
                    'HFlip': torchvision.transforms.RandomHorizontalFlip(p=1),
                    'VFlip': torchvision.transforms.RandomVerticalFlip(p=1),
                }
                for (augmenter_name, augmenter) in augmenters.items():
                    resizes_son = {
                        'bilinear': torchvision.transforms.Resize((int(image_downscale.shape[1]/scale_son),
                                                                   int(image_downscale.shape[2]/scale_son)),
                                                                 interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                 antialias=True),
                        'nearest': torchvision.transforms.Resize((int(image_downscale.shape[1]/scale_son),
                                                                  int(image_downscale.shape[2]/scale_son)),
                                                                 interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                                 antialias=True),
                        'nearest_exact': torchvision.transforms.Resize((int(image_downscale.shape[1]/scale_son),
                                                                        int(image_downscale.shape[2]/scale_son)),
                                                                 interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
                                                                 antialias=True),
                        'bicubic': torchvision.transforms.Resize((int(image_downscale.shape[1]/scale_son),
                                                                  int(image_downscale.shape[2]/scale_son)),
                                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                                 antialias=True),  
                    }
                    resize_as = random.choice(list(resizes_son.values()))
                    image_downscale_son = resize_as(image_downscale)
                    img_aug_father = augmenter(image_downscale)
                    img_aug_son = augmenter(image_downscale_son)
                    self.images_hr_fathers.append(img_aug_father)
                    self.images_lr_sons.append(img_aug_son)

    def __len__(self):
        return len(self.images_hr_fathers)
    
    def __getitem__(self, index):
        data_source = self.images_lr_sons[index]
        data_target = self.images_hr_fathers[index]
        return data_source, data_target                    
        

class data_images_lrsonshrfather_realsr_x2(data_images_lrsonshrfather):
    def __init__(self,
                 lr_path = './data/RealSR(V3)/canon/train/2',
                 hr_path = './data/RealSR(V3)/canon/train/2',
                 lr_mask = 'LR',
                 hr_mask = 'HR',
                 scale=2,
                 scales = [2,3,4]
                 ):
        super(data_images_lrsonshrfather_realsr_x2, self).__init__(lr_path=lr_path, hr_path=hr_path, lr_mask=lr_mask, hr_mask=hr_mask, scale=scale, scales=scales)


class data_patches_custom_canoneosr_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/canoneosr/lr_x2',
                 target_path = './data/canoneosr/lr_x2',
                 ref_path = './data/canoneosr/hr',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=None, target_mask=None, ref_mask=None, limit=None, refastarget=False):
        super(data_patches_custom_canoneosr_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)


class data_patches_realsr_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = ['./data/RealSR(V3)/canon/train/2', './data/RealSR(V3)/Nikon/train/2'],
                 target_path = ['./data/RealSR(V3)/canon/train/2', './data/RealSR(V3)/Nikon/train/2'],
                 ref_path = ['./data/RealSR(V3)/canon/train/2', './data/RealSR(V3)/Nikon/train/2'],
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR', limit=None, refastarget=False):
        super(data_patches_realsr_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, normalize=normalize, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)

class data_patches_realsr_canon_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = ['./data/RealSR(V3)/canon/train/2'],
                 target_path = ['./data/RealSR(V3)/canon/train/2'],
                 ref_path = ['./data/RealSR(V3)/canon/train/2'],
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR', limit=None, refastarget=False):
        super(data_patches_realsr_canon_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, normalize=normalize, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)

class data_patches_realsr_canon_x2_test(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = ['./data/RealSR(V3)/canon/test/2'],
                 target_path = ['./data/RealSR(V3)/canon/test/2'],
                 ref_path = ['./data/RealSR(V3)/canon/test/2'],
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR', limit=None, refastarget=False):
        super(data_patches_realsr_canon_x2_test, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, normalize=normalize, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)

class data_patches_realsr_x3_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = ['./data/RealSR(V3)/canon/train/3', './data/RealSR(V3)/Nikon/train/3'],
                 target_path = ['./data/RealSR(V3)/canon/train/3', './data/RealSR(V3)/Nikon/train/3'],
                 ref_path = ['./data/RealSR(V3)/canon/train/3', './data/RealSR(V3)/Nikon/train/3'],
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR', limit=None, refastarget=False):
        super(data_patches_realsr_x3_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)

class data_patches_realsr_x4_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = ['./data/RealSR(V3)/canon/train/4', './data/RealSR(V3)/Nikon/train/4'],
                 target_path = ['./data/RealSR(V3)/canon/train/4', './data/RealSR(V3)/Nikon/train/4'],
                 ref_path = ['./data/RealSR(V3)/canon/train/4', './data/RealSR(V3)/Nikon/train/4'],
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR', limit=None, refastarget=False):
        super(data_patches_realsr_x4_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)        

class data_patches_imagenet(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/imagenet/train',
                 target_path = './data/imagenet/train',
                 ref_path = './data/imagenet/train',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        super(data_patches_imagenet, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)


class data_patches_div2k_unknown_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_unknown/X2',
                 target_path = './data/div2k/DIV2K_train_LR_unknown/X2',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        super(data_patches_div2k_unknown_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, lmit=limit, refastarget=refastarget)
        
class data_patches_div2k_unknown_x3_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_unknown/X3',
                 target_path = './data/div2k/DIV2K_train_LR_unknown/X3',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        super(data_patches_div2k_unknown_x3_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)
        
class data_patches_div2k_unknown_x4_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_unknown/X4',
                 target_path = './data/div2k/DIV2K_train_LR_unknown/X4',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        super(data_patches_div2k_unknown_x4_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)        

class data_patches_div2k_bicubic_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_bicubic/X2',
                 target_path = './data/div2k/DIV2K_train_LR_bicubic/X2',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        super(data_patches_div2k_bicubic_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)
        
class data_patches_div2k_bicubic_x3_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_bicubic/X3',
                 target_path = './data/div2k/DIV2K_train_LR_bicubic/X3',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        super(data_patches_div2k_bicubic_x3_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)
        
class data_patches_div2k_bicubic_x4_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_bicubic/X4',
                 target_path = './data/div2k/DIV2K_train_LR_bicubic/X4',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None, refastarget=False):
        super(data_patches_div2k_bicubic_x4_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit, refastarget=refastarget)



# datasets for benchmarcks
        
class data_bsd100_x2(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/B100/LR_bicubic/X2',
                 ref_path = './data/benchmark/B100/HR',
                 ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_bsd100_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)

class data_bsd100_x3(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/B100/LR_bicubic/X3',
                 ref_path = './data/benchmark/B100/HR',
                 ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_bsd100_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)

class data_bsd100_x4(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/B100/LR_bicubic/X4',
                 ref_path = './data/benchmark/B100/HR',
                 ref_scale=4, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_bsd100_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)


class data_manga109_x2(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Manga109/LR_bicubic/X2',
                 ref_path = './data/benchmark/Manga109/HR',
                 ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_manga109_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_manga109_x3(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Manga109/LR_bicubic/X3',
                 ref_path = './data/benchmark/Manga109/HR',
                 ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_manga109_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_manga109_x4(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Manga109/LR_bicubic/X4',
                 ref_path = './data/benchmark/Manga109/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_manga109_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        
        
class data_set5_x2(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set5/LR_bicubic/X2',
                 ref_path = './data/benchmark/Set5/HR',
                 ref_scale=2, normalize=False, source_mask=False, ref_mask=False):
        super(data_set5_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_set5_x3(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set5/LR_bicubic/X3',
                 ref_path = './data/benchmark/Set5/HR',
                 ref_scale=3, normalize=False, source_mask=False, ref_mask=False):
        super(data_set5_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_set5_x4(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set5/LR_bicubic/X4',
                 ref_path = './data/benchmark/Set5/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_set5_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

        
class data_set14_x2(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set14/LR_bicubic/X2',
                 ref_path = './data/benchmark/Set14/HR',
                 ref_scale=2, normalize=False, source_mask=False, ref_mask=False):
        super(data_set14_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)

class data_set14_x3(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set14/LR_bicubic/X3',
                 ref_path = './data/benchmark/Set14/HR',
                 ref_scale=3, normalize=False, source_mask=False, ref_mask=False):
        super(data_set14_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_set14_x4(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set14/LR_bicubic/X4',
                 ref_path = './data/benchmark/Set14/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_set14_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_urban100_x2(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Urban100/LR_bicubic/X2',
                 ref_path = './data/benchmark/Urban100/HR',
                 ref_scale=2, normalize=False, source_mask=False, ref_mask=False):
        super(data_urban100_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)

class data_urban100_x3(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Urban100/LR_bicubic/X3',
                 ref_path = './data/benchmark/Urban100/HR',
                 ref_scale=3, normalize=False, source_mask=False, ref_mask=False):
        super(data_urban100_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_urban100_x4(image_dataset_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Urban100/LR_bicubic/X4',
                 ref_path = './data/benchmark/Urban100/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_urban100_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        


# div2k valid
class data_patches_div2k_unknown_x2_valid(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_valid_LR_unknown/X2',
                 target_path = './data/div2k/DIV2K_valid_LR_unknown/X2',
                 ref_path = './data/div2k/DIV2K_valid_HR',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None):
        super(data_patches_div2k_unknown_x2_valid, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, lmit=limit)
        
class data_patches_div2k_unknown_x3_valid(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_valid_LR_unknown/X3',
                 target_path = './data/div2k/DIV2K_valid_LR_unknown/X3',
                 ref_path = './data/div2k/DIV2K_valid_HR',
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None):
        super(data_patches_div2k_unknown_x3_valid, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit)
        
class data_patches_div2k_unknown_x4_valid(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_valid_LR_unknown/X4',
                 target_path = './data/div2k/DIV2K_valid_LR_unknown/X4',
                 ref_path = './data/div2k/DIV2K_valid_HR',
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None):
        super(data_patches_div2k_unknown_x4_valid, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit)        

class data_patches_div2k_bicubic_x2_valid(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_valid_LR_bicubic/X2',
                 target_path = './data/div2k/DIV2K_valid_LR_bicubic/X2',
                 ref_path = './data/div2k/DIV2K_valid_HR',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None):
        super(data_patches_div2k_bicubic_x2_valid, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit)
        
class data_patches_div2k_bicubic_x3_valid(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_valid_LR_bicubic/X3',
                 target_path = './data/div2k/DIV2K_valid_LR_bicubic/X3',
                 ref_path = './data/div2k/DIV2K_valid_HR',
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None):
        super(data_patches_div2k_bicubic_x3_valid, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit)
        
class data_patches_div2k_bicubic_x4_valid(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_valid_LR_bicubic/X4',
                 target_path = './data/div2k/DIV2K_valid_LR_bicubic/X4',
                 ref_path = './data/div2k/DIV2K_valid_HR',
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask=False, target_mask=False, ref_mask=False, limit=None):
        super(data_patches_div2k_bicubic_x4_valid, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask, limit=limit)        