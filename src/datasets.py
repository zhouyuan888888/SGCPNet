from __future__ import print_function, division

import os
import pdb
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class Pad(object):

    def __init__(self, size, img_val, msk_val):

        if isinstance(size, int):
            self.size_h = size
            self.size_w = size
        else:
            self.size_h=size[0]
            self.size_w =size[1]

        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size_h - h) + 1)// 2, 0, 1e6))
        w_pad = int(np.clip(((self.size_w - w) + 1)// 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        image = np.stack([np.pad(image[:,:,c], pad,
                         mode='constant',
                         constant_values=self.img_val[c]) for c in range(3)], axis=2)
        mask = np.pad(mask, pad, mode='constant', constant_values=self.msk_val)
        return {'image': image, 'mask': mask}

class RandomCrop(object):

    def __init__(self, crop_size):
        if isinstance(crop_size, int):

            if crop_size % 2 != 0:
                crop_size -= 1
            self.crop_h = crop_size
            self.crop_w = crop_size
        else:
            self.crop_h=crop_size[0]
            self.crop_w = crop_size[1]
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        new_h = min(h, self.crop_h)
        new_w = min(w, self.crop_w)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top: top + new_h,
                        left: left + new_w]
        mask = mask[top: top + new_h,
                    left: left + new_w]
        return {'image': image, 'mask': mask}

class ResizeShorterScale(object):

    def __init__(self, shorter_side, low_scale, high_scale):
        assert isinstance(shorter_side, int)
        self.shorter_side = shorter_side
        self.low_scale = low_scale
        self.high_scale = high_scale

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        min_side = min(image.shape[:2])
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if min_side * scale < self.shorter_side:
            scale = (self.shorter_side * 1. / min_side)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'mask' : mask}

class RandomMirror(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return {'image': image, 'mask' : mask}

class Normalise(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        return {'image': (self.scale * image - self.mean) / self.std, 'mask' : sample['mask']}

class ToTensor(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class ValScale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        image = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        return {"image":image, "mask":mask}

class NYUDataset(Dataset):

    def __init__(
        self, data_file, data_dir, val_scale=None, transform_trn=None, transform_val=None
        ):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist)]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        if img_name != msk_name:
            assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample = {'image': image, 'mask': mask}
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        return sample

class CityscapesDataset(Dataset):

    def __init__(self, data_file, data_dir, val_scale=None, transform_trn=None, transform_val=None):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split(' '), datalist)]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

        if val_scale is None:
            self.val_scale = None
        else:
            self.val_scale = ValScale(val_scale)


    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr

        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))

        #cv2.imwrite("/home/zhouyuan/1.png", image)
        ##---resize for 1024x1024 input images---##
        #image = Image.fromarray(image)
        #mask = Image.fromarray(mask)
        #image=np.array(image.resize((1536, 768)))
        #mask = np.array(mask.resize((1536, 768)))


        if img_name != msk_name:
            assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample = {'image': image, 'mask': mask}
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':

            #if self.val_scale is not None:
                #sample = self.val_scale(sample)
            if self.transform_val:
                sample = self.transform_val(sample)
        return sample

class ContextDataset(Dataset):


    def __init__(self, data_file, data_dir, val_scale=None, transform_trn=None, transform_val=None):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split(' '), datalist)]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

        if val_scale is None:
            self.val_scale = None
        else:
            self.val_scale = ValScale(val_scale)


    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = self.root_dir + self.datalist[idx][0]
        msk_name = self.root_dir + self.datalist[idx][1].strip("\r")
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        if img_name != msk_name:
            assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample = {'image': image, 'mask': mask}
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':

            #if self.val_scale is not None:
                #sample = self.val_scale(sample)
            if self.transform_val:
                sample = self.transform_val(sample)
        return sample


class CamvidDataset(Dataset):

    def __init__(self, data_file, data_dir, val_scale=None, transform_trn=None, transform_val=None):

        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split(' '), datalist)]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

        if val_scale is None:
            self.val_scale = None
        else:
            self.val_scale = ValScale(val_scale)


    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        if img_name != msk_name:
            assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample = {'image': image, 'mask': mask}
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':

            #if self.val_scale is not None:
                #sample = self.val_scale(sample)
            if self.transform_val:
                sample = self.transform_val(sample)
        return sample