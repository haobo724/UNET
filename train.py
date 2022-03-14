import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from mutil_train import unet_train,mutil_train
from pytorch_lightning.loggers import TensorBoardLogger
from utils import (
    get_testloaders,
get_loaders_multi

)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMAGE_HEIGHT = 274  # 1096 originally  0.25
# IMAGE_WIDTH = 484  # 1936 originally 164 290
IMAGE_HEIGHT = 256 # 1096 originally  0.25
IMAGE_WIDTH = 256 # 1936 originally
# print(IMAGE_HEIGHT,IMAGE_WIDTH)
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/train_set/"
TRAIN_MASK_DIR = "data/train_set_mask/"
VAL_IMG_DIR = "data/train_set/"
VAL_MASK_DIR = "data/train_set_mask/"
test_dir = r"C:\Users\94836\Desktop\test_data"
test_maskdir =r"C:\Users\94836\Desktop\test_data"
def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--mode_size', type=int, default=64)
    parser.add_argument("--model", type=str, default='Unet')

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

    # model = unet_train(hparams=vars(args)).cuda()
    model = mutil_train(hparams=vars(args)).cuda()

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
        seed=1111
    )
    if args.mode_size == 32:
        name = 'S'
    elif args.mode_size == 16:
        name = 'XS'
    else:
        name = 'M'
    ckpt_callback = ModelCheckpoint(
        monitor='val_Iou',
        save_top_k=2,
        mode='max',
        filename='{epoch:02d}-{val_Iou:.2f}',
        save_last=True

    )
    # logger = TensorBoardLogger(save_dir=os.path.join('.', 'lightning_logs'))
    trainer = pl.Trainer.from_argparse_args(args, check_val_every_n_epoch=3, log_every_n_steps=5,
                                            callbacks=[ckpt_callback])


    trainer.fit(model, train_loader, val_loader)

    print('THE END')


def test():
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
    trainer = pl.Trainer()
    test_loader = get_testloaders(test_dir,
                    test_maskdir,
                    1,
                    val_transforms,
                    4,
                    pin_memory=True,)
    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}')
    model = mutil_train.load_from_checkpoint(r'F:\semantic_segmentation_unet\last.ckpt', hparams=vars(args))
    trainer.test(model, test_loader)

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
