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
    def __init__(self, source_path, ref_path, ref_scale=2, normalize=False, source_mask=False, ref_mask=False):
        self.source_path = source_path
        self.ref_path = ref_path
        self.ref_scale = ref_scale
        self.normalize = normalize
        self.source_mask = source_mask
        self.ref_mask = ref_mask
        self.images_source = []
        self.images_ref = []
        self.images_source_img = []
        self.images_ref_img = []
                
        for filename in tqdm(os.listdir(self.source_path)):
            if self.source_mask:
                if filename.find(self.source_mask)>0:
                    self.images_source.append(os.path.join(self.source_path, filename))
            else:
                self.images_source.append(os.path.join(self.source_path, filename))


        for filename in tqdm(os.listdir(self.ref_path)):
            if self.ref_mask:
                if filename.find(self.target_mask)>0:
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
    def __init__(self, source_path, target_path, ref_path, patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        self.source_path = source_path
        self.target_path = target_path
        self.ref_path = ref_path
        self.patch_size = patch_size
        self.ref_scale = ref_scale
        self.normalize = normalize
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.ref_mask = ref_mask
        self.images_source = []
        self.images_target = []
        self.images_ref = []
        self.images_source_img = []
        self.images_target_img = []
        self.images_ref_img = []
        self.images_source_patches = []
        self.images_target_patches = []
        self.images_ref_patches = []
        
                
        for filename in os.listdir(self.source_path):
            if self.source_mask:
                if filename.find(self.source_mask)>0:
                    self.images_source.append(os.path.join(self.source_path, filename))
            else:
                self.images_source.append(os.path.join(self.source_path, filename))

        for filename in os.listdir(self.target_path):
            if self.target_mask:
                if filename.find(self.target_mask)>0:
                    self.images_target.append(os.path.join(self.target_path, filename))
            else:
                self.images_target.append(os.path.join(self.target_path, filename))

        for filename in os.listdir(self.ref_path):
            if self.ref_mask:
                if filename.find(self.ref_mask)>0:
                    self.images_ref.append(os.path.join(self.ref_path, filename))
            else:
                self.images_ref.append(os.path.join(self.ref_path, filename))
                
                
        image_transforms = []
        image_transforms.append(torchvision.transforms.v2.ToImage())
        image_transforms.append(torchvision.transforms.v2.ToDtype(torch.float32, scale=True))
        self.transform = torchvision.transforms.Compose(image_transforms)
        
        for image in tqdm(self.images_source):
            img = read_image(image)
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
                
        for image in tqdm(self.images_target):
            img = read_image(image)
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
                
        for image in tqdm(self.images_ref):
            img = read_image(image)
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

                
    def __len__(self):
        return len(self.images_source)
    
    def __getitem__(self, index):
        data_source = self.images_source_patches[index]
        data_target = self.images_target_patches[index]
        return data_source, data_target

class data_patches_realsr_canon_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/RealSR(V3)/canon/train/2',
                 target_path = './data/RealSR(V3)/canon/train/2',
                 ref_path = './data/RealSR(V3)/canon/train/2',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR'):
        super(data_patches_realsr_canon_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)

class data_patches_realsr_canon_x3_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/RealSR(V3)/canon/train/3',
                 target_path = './data/RealSR(V3)/canon/train/3',
                 ref_path = './data/RealSR(V3)/canon/train/3',
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR'):
        super(data_patches_realsr_canon_x3_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)

class data_patches_realsr_canon_x4_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/RealSR(V3)/canon/train/4',
                 target_path = './data/RealSR(V3)/canon/train/4',
                 ref_path = './data/RealSR(V3)/canon/train/4',
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask='LR', target_mask='LR', ref_mask='HR'):
        super(data_patches_realsr_canon_x4_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)        

class data_patches_imagenet(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/imagenet/train',
                 target_path = './data/imagenet/train',
                 ref_path = './data/imagenet/train',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_patches_imagenet, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)


class data_patches_div2k_unknown_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_unknown/X2',
                 target_path = './data/div2k/DIV2K_train_LR_unknown/X2',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_patches_div2k_unknown_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)
        
class data_patches_div2k_unknown_x3_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_unknown/X3',
                 target_path = './data/div2k/DIV2K_train_LR_unknown/X3',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_patches_div2k_unknown_x3_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)
        
class data_patches_div2k_unknown_x4_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_unknown/X4',
                 target_path = './data/div2k/DIV2K_train_LR_unknown/X4',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_patches_div2k_unknown_x4_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)        

class data_patches_div2k_bicubic_x2_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_bicubic/X2',
                 target_path = './data/div2k/DIV2K_train_LR_bicubic/X2',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=2, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_patches_div2k_bicubic_x2_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)
        
class data_patches_div2k_bicubic_x3_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_bicubic/X3',
                 target_path = './data/div2k/DIV2K_train_LR_bicubic/X3',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=3, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_patches_div2k_bicubic_x3_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)
        
class data_patches_div2k_bicubic_x4_train(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/div2k/DIV2K_train_LR_bicubic/X4',
                 target_path = './data/div2k/DIV2K_train_LR_bicubic/X4',
                 ref_path = './data/div2k/DIV2K_train_HR',
                 patch_size=(256,256), ref_scale=4, normalize=False, source_mask=False, target_mask=False, ref_mask=False):
        super(data_patches_div2k_bicubic_x4_train, self).__init__(source_path, target_path, ref_path, patch_size=patch_size, ref_scale=ref_scale, source_mask=source_mask, target_mask=target_mask, ref_mask=ref_mask)



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

class data_manga109_x4(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Manga109/LR_bicubic/X4',
                 ref_path = './data/benchmark/Manga109/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_manga109_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        
        
class data_set5_x2(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set5/LR_bicubic/X2',
                 ref_path = './data/benchmark/Set5/HR',
                 ref_scale=2, normalize=False, source_mask=False, ref_mask=False):
        super(data_set5_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_set5_x3(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set5/LR_bicubic/X3',
                 ref_path = './data/benchmark/Set5/HR',
                 ref_scale=3, normalize=False, source_mask=False, ref_mask=False):
        super(data_set5_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_set5_x4(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set5/LR_bicubic/X4',
                 ref_path = './data/benchmark/Set5/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_set5_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

        
class data_set14_x2(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set14/LR_bicubic/X2',
                 ref_path = './data/benchmark/Set14/HR',
                 ref_scale=2, normalize=False, source_mask=False, ref_mask=False):
        super(data_set14_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)

class data_set14_x3(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set14/LR_bicubic/X3',
                 ref_path = './data/benchmark/Set14/HR',
                 ref_scale=3, normalize=False, source_mask=False, ref_mask=False):
        super(data_set14_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_set14_x4(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Set14/LR_bicubic/X4',
                 ref_path = './data/benchmark/Set14/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_set14_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_urban100_x2(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Urban100/LR_bicubic/X2',
                 ref_path = './data/benchmark/Urban100/HR',
                 ref_scale=2, normalize=False, source_mask=False, ref_mask=False):
        super(data_urban100_x2, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)

class data_urban100_x3(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Urban100/LR_bicubic/X3',
                 ref_path = './data/benchmark/Urban100/HR',
                 ref_scale=3, normalize=False, source_mask=False, ref_mask=False):
        super(data_urban100_x3, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        

class data_urban100_x4(image_dataset_patches_from_file):
    def __init__(self,
                 source_path = './data/benchmark/Urban100/LR_bicubic/X4',
                 ref_path = './data/benchmark/Urban100/HR',
                 ref_scale=4, normalize=False, source_mask=False, ref_mask=False):
        super(data_urban100_x4, self).__init__(source_path, ref_path, ref_scale=ref_scale, source_mask=source_mask, ref_mask=ref_mask)        
        