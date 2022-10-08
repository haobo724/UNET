from argparse import ArgumentParser
import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint

from mutil_train import mutil_train
from utils import (
    add_training_args,
    get_loaders_multi,
    cal_std_mean

)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

PIN_MEMORY = True
TRAIN_IMG_DIR = "F:\semantic_segmentation_unet\data\All_clinic"
TRAIN_MASK_DIR = "F:\semantic_segmentation_unet\data\All_clinic_mask"
VAL_IMG_DIR = TRAIN_IMG_DIR
VAL_MASK_DIR = TRAIN_MASK_DIR
test_dir = r"F:\semantic_segmentation_unet\data\test_new"
test_maskdir = r"F:\semantic_segmentation_unet\data\test_new_mask"


def main(args):
    pl.seed_everything(1211)
    mean_value, std_value = cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
    torch.cuda.empty_cache()
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
            A.ColorJitter(brightness=0.3,contrast=0.3,saturation=0,hue=0, p=0.3),
            A.Rotate(limit=10, p=0.2),
            A.HorizontalFlip(p=0.2),
            A.Normalize(
                mean=mean_value,
                std=std_value,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=mean_value,
                std=std_value,
            ),
            ToTensorV2(),
        ],
    )


    model = mutil_train(hparams=vars(args)).cuda()
    model_name = model.get_model_info()
    train_loader, val_loader = get_loaders_multi(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        args.batch_size,
        train_transform,
        val_transforms,
        args.worker,
        PIN_MEMORY,
        seed=args.seed
    )

    print('Train images:', len(train_loader) * args.batch_size)
    print('Validation  images:', len(val_loader) * args.batch_size)


    prefix = model_name + '_' + str(IMAGE_WIDTH) + '_' + str(IMAGE_HEIGHT)
    ckpt_callback = ModelCheckpoint(
        monitor='val_Iou',
        save_top_k=1,
        mode='max',
        filename='{}'.format(prefix) + '-{epoch:02d}-{val_Iou:.2f}',

        save_last=True

    )

    path = 'F:\semantic_segmentation_unet\clinic_exper\epoch=179-val_Iou=0.62.ckpt'
    if args.Continue == True:

        args.max_epochs *= 3
        trainer = pl.Trainer.from_argparse_args(args, resume_from_checkpoint=path, check_val_every_n_epoch=3,
                                                log_every_n_steps=5,
                                                callbacks=[ckpt_callback])
        print('Continue train from %s'.format({path}))
    else:
        trainer = pl.Trainer.from_argparse_args(args, check_val_every_n_epoch=3,
                                                log_every_n_steps=5,
                                                callbacks=[ckpt_callback])

    trainer.fit(model, train_loader, val_loader)

    print('THE END')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = mutil_train.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
