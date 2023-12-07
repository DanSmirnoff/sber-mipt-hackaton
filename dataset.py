import os
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

class LitographyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, aug=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.aug = aug

        self.img_paths = []
        self.mask_paths = []

        for fname in os.listdir(self.img_dir):
            path = self.img_dir+fname

            if os.path.isfile(path):
                self.img_paths.append(path)

        for fname in os.listdir(self.mask_dir):
            path = self.mask_dir+fname

            if os.path.isfile(path):
                self.mask_paths.append(path)


        self.length = len(self.img_paths)


    def __len__(self):
        return self.length

    def _load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def _prepare_sample(self, image):
        return np.array(image)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = self._load_sample(img_path)
        image = self._prepare_sample(image)

        mask = self._load_sample(mask_path)
        mask = self._prepare_sample(mask)


        if self.aug:
            augmented = self.aug(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.from_numpy(image).to(dtype=torch.float32)/255
        mask = torch.from_numpy(mask).to(dtype=torch.float32)/255

        return image[None, :, :], mask[None, :, :]