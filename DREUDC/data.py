from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import skimage.measure
import random
import os
import glob
import cv2
from skimage import io


# Dataset class
class ImageDataset_raw2rgb(Dataset):
    def __init__(self, image_dirs, panel_type):
        # image_dirs : has HQ/LQ directory
        # image_files : noisy image
        # targets : clean image

        self.image_dirs = image_dirs
        self.panel_type = panel_type

        if self.panel_type == 'axon':
            self.udc_files_dirs = image_dirs + 'raw/test/axon/*.png'
        else:
            self.udc_files_dirs = image_dirs + 'raw/test/zfold/*.png'

        self.clean_files_dirs = image_dirs + 'rgb/test/clean/*.png'

        self.udc_files = glob.glob(self.udc_files_dirs)
        self.clean_files = glob.glob(self.clean_files_dirs)

        self.udc_files.sort()
        self.clean_files.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.udc_files)

    def __getitem__(self, idx):
        img_name = self.udc_files[idx]
        target_name = self.clean_files[idx]

        image = (cv2.imread(img_name, cv2.IMREAD_ANYDEPTH).astype(np.float32)) / 65535
        target = (cv2.cvtColor(cv2.imread(target_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)) / 255

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        sample = image, target

        return sample
