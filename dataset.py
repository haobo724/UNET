import os

import cv2
import imageio
import numpy as np
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, imgs, masks, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # self.images = os.listdir(image_dir)
        # self.masks = os.listdir(mask_dir)
        self.images = imgs
        self.masks = masks

        print(len(self.images))
        print(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        img = imageio.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = cv2.imread(mask_path)[..., 0].astype(np.float32)
        if np.max(mask) != 1:
            if len(np.unique(mask)) == 2:
                low, high = np.unique(mask)
                mask[mask == high] = 1.0
                mask[mask == low] = 0.0
            else:
                mask[mask > 0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class CarvanaDataset_multi(Dataset):
    def __init__(self, image_dir, mask_dir, imgs, masks, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = imgs
        self.masks = masks

        print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        img = imageio.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = cv2.imread(mask_path)[..., 0].astype(np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class LeafData(Dataset):

    def __init__(self,
                 data,
                 transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # import
        # path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        img = imageio.imread(self.data[idx])
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = np.array(imageio.imread(self.data[idx]))
        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image
