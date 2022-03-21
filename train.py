import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
# from matplotlib import pyplot as plt

from mutil_train import unet_train, mutil_train
# from pytorch_lightning.loggers import TensorBoardLogger
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
IMAGE_HEIGHT = 256  # 1096 originally  0.25
IMAGE_WIDTH = 256  # 1936 originally
# print(IMAGE_HEIGHT,IMAGE_WIDTH)
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/train_set/"
TRAIN_MASK_DIR = "data/train_set_mask/"
VAL_IMG_DIR = "data/train_set/"
VAL_MASK_DIR = "data/train_set_mask/"
test_dir = r"C:\Users\94836\Desktop\test_data"
test_maskdir = r"C:\Users\94836\Desktop\test_data"


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--mode_size', type=int, default=64)
    parser.add_argument("--model", type=str, default='Unet')

    return parser


def main():
    pl.seed_everything(1111)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ColorJitter(brightness=0.3, hue=0.3, p=0.3),
            A.Rotate(limit=5, p=1.0),
            # A.HorizontalFlip(p=0.3),
            # A.VerticalFlip(p=0.2),
            A.Normalize(
                mean=(0.617,
                      0.6087,
                      0.6254),
                std=(0.208,
                     0.198,
                     0.192),
                max_pixel_value=255.0,

            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.617,
                      0.6087,
                      0.6254),
                std=(0.208,
                0.198,
                0.192),
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
                # mean=[0.0, 0.0, 0.0],
                # std=[1.0, 1.0, 1.0],
                mean=(0.617,
                      0.6087,
                      0.6254),
                std=(0.208,
                     0.198,
                     0.192),
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
                                  pin_memory=True, )
    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}')
    model = mutil_train.load_from_checkpoint(r'.\last.ckpt', hparams=vars(args))
    trainer.test(model, test_loader)

def infer_multi(model):
    model = mutil_train.load_from_checkpoint(model)
    infer_xform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                # mean=(0.617,
                #       0.6087,
                #       0.6254),
                # std=(0.208,
                #      0.198,
                #      0.192),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    codec = cv2.VideoWriter_fourcc(*'MJPG')
    frameSize_s = (512, 256)  # 指定窗口大小

    # video_path = './video/breast.avi'
    videos = glob.glob('./video/clinical/*.avi')
    for video_path in videos:
    # video_path = './video/c4.avi'
        cap = cv2.VideoCapture(video_path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        name = 'mask_'+video_path.split('\\')[-1]
        out_path = os.path.join('./video/',name)
        print(out_path)
        out = cv2.VideoWriter(out_path, codec,5, frameSize_s)
        with torch.no_grad():

            with tqdm(total=total_frames) as pbar:

                while True:
                    ret, frame = cap.read()
                    if ret:
                        # print(frame.shape)
                        pbar.update(1)
                        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        input = frame
                        # resize_xform = A.Compose(
                        #     [
                        #         A.Resize(height=input.shape[0], width=input.shape[1]),
                        #
                        #         ToTensorV2(),
                        #     ],
                        # )
                        input = infer_xform(image=input)
                        x = input["image"].cuda()

                        x = torch.unsqueeze(x, dim=0)
                        model.eval()
                        y_hat = model(x)
                        preds = torch.softmax(y_hat, dim=1)

                        pred = preds.argmax(dim=1).float().cpu()
                        img = np.stack([pred[0] for _ in range(3)], axis=-1)
                        img = mapping_color(img)
                        # temp = np.array(torch.movedim(x[0].cpu(), 0, 2) * 255)
                        temp = cv2.resize(frame,(IMAGE_HEIGHT,IMAGE_WIDTH))
                        concat = np.hstack([img, temp]).astype(np.uint8)
                        concat = cv2.cvtColor(concat,cv2.COLOR_BGR2RGB)
                        # concat[...,0],concat[...,2]= concat[...,2],concat[...,0]
                        # print(concat.shape)
                        # concat = cv2.cvtColor(concat,cv2.COLOR_BGR2RGB)
                        # cv2.imshow('test',concat)
                        # cv2.waitKey(15)
                        out.write(concat)
                    else:
                        break
                    if cv2.waitKey(1) == ord('q'):
                        break
            cap.release()
            out.release()
def mapping_color(img):
    '''
    自己写的，速度快不少，但要自己规定colormap，也可以把制定colormap拿出来单独用randint做，
    但是不能保证一个series里每次运行生成的colormap都一样，或许可以用种子点？
    反正类少还是可以考虑用这个
            '''
    color_map = [[247, 251, 255], [171, 207, 209], [55, 135, 192]]
    for label in range(3):
        cord_1 = np.where(img[..., 0] == label)
        img[cord_1[0], cord_1[1], 0] = color_map[label][0]
        img[cord_1[0], cord_1[1], 1] = color_map[label][1]
        img[cord_1[0], cord_1[1], 2] = color_map[label][2]
    return img.astype(int)


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
    # model = './epoch=95-val_Iou=0.60.ckpt'
    model = './last.ckpt'
    infer_multi(model)
