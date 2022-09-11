import glob
import os
from argparse import ArgumentParser
import imageio
import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from mutil_train import mutil_train
from utils import (
    add_training_args,
    get_loaders_multi,
    cal_std_mean

)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 480  # 1096 originally  0.25
IMAGE_WIDTH = 640  # 1936 originally 164 290

PIN_MEMORY = True
TRAIN_IMG_DIR = "data/elbows/"
TRAIN_MASK_DIR = "data/elbows_mask/"
VAL_IMG_DIR = TRAIN_IMG_DIR
VAL_MASK_DIR = TRAIN_MASK_DIR
test_dir = r"testdata/"
test_maskdir = r"testdata/"


def main():
    pl.seed_everything(1211)
    mean_value, std_value = cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
            # A.ColorJitter(brightness=0.3, hue=0.2, p=0.3),
            A.Rotate(limit=5, p=0.3),
            A.HorizontalFlip(p=0.2),            # A.VerticalFlip(p=0.2),
            A.Normalize(
                mean=mean_value,
                std=std_value,
                # max_pixel_value=255.0,

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
                # max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    parser = ArgumentParser()
    parser = add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = mutil_train.add_model_specific_args(parser)
    args = parser.parse_args()

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
        seed=1111
    )

    print('Train images:', len(train_loader) * args.batch_size)
    print('Validation  images:', len(val_loader) * args.batch_size)

    if args.model != 'Unet':
        ckpt_callback = ModelCheckpoint(
            monitor='val_Iou',
            save_top_k=2,
            mode='max',
            filename='{Unetppepoch:02d}-{val_Iou:.2f}',
            save_last=True

        )
    else:
        prefix = model_name + '_' + str(IMAGE_WIDTH) + '_' + str(IMAGE_HEIGHT)
        ckpt_callback = ModelCheckpoint(
            monitor='val_Iou',
            save_top_k=2,
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



def infer_multi(model):
    parser = ArgumentParser()
    parser = add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = mutil_train.add_model_specific_args(parser)
    args = parser.parse_args()

    model = mutil_train.load_from_checkpoint(model, hparams=vars(args))
    mean_v, std_v = cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
    infer_xform = A.Compose(
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

    frameSize_s = (IMAGE_WIDTH, IMAGE_HEIGHT)  # 指定窗口大小

    # videos = ['.\\video\\breast.avi']
    # videos = glob.glob('./video/*.avi')
    videos = glob.glob(r'F:\semantic_segmentation_unet\video\*.mp4')

    for video_path in videos:
        # video_path = './video/c4.avi'
        cap = cv2.VideoCapture(video_path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        name = 'output_' + video_path.split('\\')[-1]
        out_path = os.path.join('./video/output/', name)
        print(out_path)
        suffix = out_path.split('.')[-1]
        if suffix == 'mp4' or suffix == 'MP4':
            codec = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            # suffix == 'avi' or suffix == 'avi':
            codec = cv2.VideoWriter_fourcc(*'MJPG')
        # imggg = imageio.imread(r'F:\semantic_segmentation_unet\data\elbows\67.jpg')
        # cv2.imshow('imggg', imggg)
        # cv2.waitKey(0)
        out = cv2.VideoWriter(out_path, codec, 15, frameSize_s)
        with torch.no_grad():

            with tqdm(total=total_frames) as pbar:

                while True:
                    ret, frame = cap.read()
                    if ret:
                        # print(frame.shape)
                        pbar.update(1)
                        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        # frame =np.rot90(frame,-2)
                        input = frame

                        input = infer_xform(image=input)
                        x = input["image"].cuda()

                        x = torch.unsqueeze(x, dim=0)
                        model.eval()
                        y_hat = model(x)
                        preds = torch.softmax(y_hat, dim=1)

                        pred = preds.argmax(dim=1).float().cpu()
                        img = np.stack([pred[0] for _ in range(3)], axis=-1)
                        img = mapping_color(img).astype(np.uint8)
                        temp = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.uint8)

                        concat = cv2.addWeighted(temp,0.5,img,0.5,0).astype(np.uint8)
                        # concat = cv2.cvtColor(concat, cv2.COLOR_BGR2RGB)
                        cv2.imshow('test', concat)
                        cv2.waitKey(1)
                        # cv2.imwrite('testtttt.tiff',img[...,0].astype(np.uint8))
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
    color_map = [[247, 251, 255], [1, 207, 209], [2, 135, 192]]
    for label in range(3):
        cord_1 = np.where(img[..., 0] == label)
        img[cord_1[0], cord_1[1], 0] = color_map[label][0]
        img[cord_1[0], cord_1[1], 1] = color_map[label][1]
        img[cord_1[0], cord_1[1], 2] = color_map[label][2]
    return img.astype(int)


if __name__ == "__main__":
    # main()
    infer_multi(r'u-resnet34_640_480-epoch=233-val_Iou=0.72.ckpt')
