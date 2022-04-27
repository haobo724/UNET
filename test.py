from albumentations.pytorch import ToTensorV2
import torch
import logging
import albumentations as A

from mutil_train import  mutil_train
from utils import (
    get_testloaders,
    add_training_args
)
import pytorch_lightning as pl
from argparse import ArgumentParser

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256  # 1096 originally  0.25
IMAGE_WIDTH = 448  # 1936 originally 164 290
# IMAGE_HEIGHT = 256  # 1096 originally  0.25
# IMAGE_WIDTH = 256  # 1936 originally
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/clinic/"
TRAIN_MASK_DIR = "data/clinic_mask/"

VAL_IMG_DIR = TRAIN_IMG_DIR
VAL_MASK_DIR = TRAIN_MASK_DIR
test_dir = r"testdata/"
test_maskdir = r"testdata/"
import cv2
from utils import cal_std_mean


def test():
    mean_v, std_v = cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
            A.Normalize(

                mean=mean_v,
                std=std_v,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    parser = ArgumentParser()
    parser = add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = mutil_train.add_model_specific_args(parser)
    args = parser.parse_args()
    trainer = pl.Trainer()
    test_loader = get_testloaders(test_dir,
                                  test_maskdir,
                                  1,
                                  val_transforms,
                                  4,
                                  pin_memory=True, )
    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}')
    model = mutil_train.load_from_checkpoint(r'model_pixel/self—res——epoch=125-val_Iou=0.75.ckpt',
                                             hparams=vars(args))
    trainer.test(model, test_loader)

if __name__ == "__main__":
    test()