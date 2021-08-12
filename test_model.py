import os

import numpy as np

import glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torchvision

from model import UNET
from utils import (
    get_loaders,
    get_testloaders

)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from train import unet_train
from argparse import ArgumentParser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 50
NUM_WORKERS = 8
IMAGE_HEIGHT = 274  # 1096 originally  0.25
IMAGE_WIDTH = 484  # 1936 originally
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def test(modeln,dir):

    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    logger = TensorBoardLogger(save_dir=os.path.join('.', 'lightning_logs'), name='my_test')
    model= unet_train.load_from_checkpoint(modeln)
    test_loader=get_testloaders(test_dir=TRAIN_IMG_DIR,test_maskdir=TRAIN_MASK_DIR,test_transform=test_transforms,batch_size=1,num_workers=4)
    tester=pl.Trainer(gpus=-1,logger=logger)
    tester.test(model,test_dataloaders=test_loader)
if __name__ == "__main__":
    modelslist = []
    for root, dirs, files in os.walk(r".\goodmodel"):
        for file in files:
            if file.endswith('.ckpt'):
                modelslist.append(os.path.join(root, file))
    print(modelslist)
    # g=infer_gui(modelslist[0])
    # image=g.forward(r'C:\Users\z00461wk\Desktop\Pressure_measure_activate tf1x\Camera_util/breast.jpg')
    # print(image.shape)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    test(modelslist[0], './testdata')