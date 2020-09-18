from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from scipy.io import loadmat


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img_nd, scale):
        img_nd = img_nd.astype(float)
        
        if len(img_nd.shape) == 2:
    
            w, h = img_nd.shape
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            pil_img = Image.fromarray(img_nd)
            pil_img = pil_img.resize((newH, newW), resample=Image.BILINEAR)
            img_snd = np.array(pil_img).astype(float)
            img_snd = np.expand_dims(img_snd, axis=2)
        
        if len(img_nd.shape) == 3:
    
            w, h, c = img_nd.shape
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            img_snd = np.arange(newW * newH * c).reshape((newW, newH, c)).astype(float)
            
            for i in range(c):
                pil_img = Image.fromarray(img_nd[:,:,i])
                pil_img = pil_img.resize((newH, newW), resample=Image.BILINEAR)
                img_snd[:,:,i] = np.array(pil_img).astype(float)

        # HWC to CHW
        img_trans = img_snd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / np.max(img_trans)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        file = loadmat(self.imgs_dir + idx)

        mask = file['hmap']
        img = file['depth']

        img = self.preprocess(img, self.scale)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
