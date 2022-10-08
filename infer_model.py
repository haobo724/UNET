import glob
import os.path
from argparse import ArgumentParser
import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from infer import add_training_args
from mutil_train import mutil_train
from start import TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH
from utils import cal_std_mean


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
    infer_multi(r'u-resnet34_640_480-epoch=233-val_Iou=0.72.ckpt')