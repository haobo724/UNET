import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from mutil_train import unet_train,mutil_train
from pytorch_lightning.loggers import TensorBoardLogger
from utils import (
    get_loaders,

)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 274  # 1096 originally  0.25
IMAGE_WIDTH = 484  # 1936 originally
# IMAGE_HEIGHT = 480  # 1096 originally  0.25
# IMAGE_WIDTH = 640  # 1936 originally
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/all_images/"
TRAIN_MASK_DIR = "data/all_masks/"
VAL_IMG_DIR = "data/all_images/"
VAL_MASK_DIR = "data/all_masks/"


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--mode_size', type=int, default=64)

    return parser




def main():
    pl.seed_everything(1111)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    parser = ArgumentParser()
    parser = add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = unet_train.add_model_specific_args(parser)
    args = parser.parse_args()

    model = unet_train(hparams=vars(args)).cuda()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        args.batch_size,
        train_transform,
        val_transforms,
        args.worker,
        PIN_MEMORY,
        seed=1111
    )
    if args.mode_size == 32:
        name = 'S'
    elif args.mode_size == 16:
        name = 'XS'
    else:
        name = 'M'
    ckpt_callback = ModelCheckpoint(
        monitor='valid_IOU',
        save_top_k=2,
        mode='max',
        filename=f'{name}' + '{epoch:02d}-{valid_IOU:02f}'

    )
    logger = TensorBoardLogger(save_dir=os.path.join('.', 'lightning_logs'))
    trainer = pl.Trainer.from_argparse_args(args, check_val_every_n_epoch=3, callbacks=[ckpt_callback], logger=logger)


    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}')


    trainer.fit(model, train_loader, val_loader)

    print('THE END')


if __name__ == "__main__":
    # modelslist = []
    # for root, dirs, files in os.walk(r".\lightning_logs"):
    #     for file in files:
    #         if file.endswith('.ckpt'):
    #             modelslist.append(os.path.join(root, file))
    # print(modelslist)
    # print(modelslist[-3])
    #
    # infer(modelslist[-3], './testdata')

    main()
