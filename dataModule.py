import glob
import os.path

import monai
import pytorch_lightning as pl
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from monai.data import DataLoader, list_data_collate
from sklearn import model_selection

from dataset import CarvanaDataset_multi


class Song_dataset_2d_with_CacheDataloder(pl.LightningDataModule):
    def __init__(self, img_dir,mask_dir, worker, batch_size, **kwargs):
        super().__init__()
        self.cache_dir = None

        self.worker = worker
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.val_ds =None
        self.train_ds=None



    def setup(self, stage: str = None):
        train_transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.ColorJitter(brightness=0.3, hue=0.3, p=0.4),
                A.Rotate(limit=5, p=1.0),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.2),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        val_transforms = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        X = glob.glob('./data/train_set/*.jpg')
        y = glob.glob('./data/train_set_mask/*.tiff')
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=123)
        train_img = []
        train_mask = []
        val_img = []
        val_mask = []

        for i, j in zip(X_test, y_test):
            i = os.path.split(i)[-1]
            j = os.path.split(j)[-1]
            val_img.append(i)
            val_mask.append(j)
        for i, j in zip(X_train, y_train):
            i = os.path.split(i)[-1]
            j = os.path.split(j)[-1]
            train_img.append(i)
            train_mask.append(j)
        print('OK')
        self.train_ds = CarvanaDataset_multi(
            image_dir=self.img_dir ,
            mask_dir=self.mask_dir ,
            transform=train_transform,
            imgs=train_img,
            masks=train_mask

        )
        self.val_ds =CarvanaDataset_multi(
            image_dir=self.img_dir ,
            mask_dir=self.mask_dir ,
            transform=val_transforms,
            imgs=val_img,
            masks=val_mask

        )

    def train_dataloader(self, cache=None):

        train_loader = DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=list_data_collate

        )
        return train_loader

    def val_dataloader(self, cache=None):

        val_loader = DataLoader(
            dataset=self.val_ds,
            batch_size=4,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=list_data_collate
        )
        return val_loader
