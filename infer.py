import os
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import glob
from caculate import calculate_eval_matrix, calculate_IoU, calculate_acc
from argparse import ArgumentParser
import pytorch_lightning as pl

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image as Image
import torchvision
from train import unet_train
import matplotlib.pyplot as plt
from model import UNET, UNET_S

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 8
IMAGE_HEIGHT = 274  # 1096 originally  0.25
IMAGE_WIDTH = 484  # 1936 originally
PIN_MEMORY = True
LOAD_MODEL = False


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--mode_size', type=int, default=64)
    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)

    return parser


def infer(models, raw_dir, sufix):
    if raw_dir is None or models is None:
        ValueError('raw_dir or model is missing')

    infer_xform = A.Compose(
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

    infer_xform2 = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
        ],
    )

    infer_data = sorted(glob.glob(os.path.join(raw_dir, "*.PNG")))
    infer_data2 = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    infer_data += infer_data2
    print(infer_data)

    folder = "saved_images_" + sufix + '//'

    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    curtime = time.time()

    model = unet_train.load_from_checkpoint(models)
    with torch.no_grad():
        model.eval()
        for index in range(len(infer_data)):
            input = np.array(Image.open(infer_data[index]), dtype=np.uint8)[..., 0:3]
            input_copy = input.copy()
            resize_xform = A.Compose(
                [
                    A.Resize(height=input.shape[0], width=input.shape[1]),

                    ToTensorV2(),
                ],
            )
            input = infer_xform(image=input)
            x = input["image"].cuda()

            x = torch.unsqueeze(x, dim=0)
            timebegin = time.time()
            # model.freeze()
            model.eval()
            y_hat = model(x)
            preds = torch.sigmoid(y_hat.squeeze())
            preds = (preds > 0.6).float()

            preds = resize_xform(image=preds.cpu().numpy())
            preds = preds["image"].numpy()
            timeend = time.time()
            post_pred = post_processing(preds)
            post_pred = np.expand_dims(post_pred, axis=0)
            preds = np.vstack((preds, preds, preds))
            preds = np.expand_dims(preds, axis=0)

            post_pred = np.vstack((post_pred, post_pred, post_pred))
            post_pred = np.expand_dims(post_pred, axis=0)

            input_copy = infer_xform2(image=input_copy)["image"]
            input_copy = np.transpose(input_copy, (2, 0, 1))
            input_copy = np.expand_dims(input_copy, axis=0)

            saved = np.vstack((input_copy, preds, post_pred))
            saved = torch.tensor(saved)
            filename = infer_data[index].split('\\')[-1]
            print(filename)
            torchvision.utils.save_image(
                saved, f"{folder}/infer_{filename}"
            )
            print(f'At {index} image used:{timeend - timebegin} s')

    end = time.time()
    print(f'Totally used:{end - curtime} s')


def metrics(models, img_dir, mask_dir, sufix='sufix'):
    if img_dir is None or models is None:
        ValueError('raw_dir or model is missing')
    filename = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
    filename_img = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_sum = []
    img_sum = []
    for mask in filename:
        mask_img = cv2.imread(mask, 0) / 255
        mask_sum.append(mask_img)
    mask_sum = np.array(mask_sum)
    # test=torch.load(models)
    # mode_size=test['state_dict']['model.ups.0.weight'].size()[0]/16
    # models['hyper_parameters'][0].update({"mode_size": int(mode_size)})

    model = unet_train.load_from_checkpoint(models)
    infer_xform = A.Compose(
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
    curtime = time.time()

    for index in range(len(filename_img)):
        input = np.array(Image.open(filename_img[index]), dtype=np.uint8)
        resize_xform = A.Compose(
            [
                A.Resize(height=input.shape[0], width=input.shape[1]),

                ToTensorV2(),
            ],
        )
        input = infer_xform(image=input)
        x = input["image"].cuda()

        x = torch.unsqueeze(x, dim=0)
        model.eval()
        y_hat = model(x)
        preds = torch.sigmoid(y_hat.squeeze())
        preds = (preds > 0.6).float()

        preds = resize_xform(image=preds.cpu().numpy())
        preds = preds["image"].numpy() * 1
        preds = preds.squeeze()
        img_sum.append(preds)
    print(time.time() - curtime)
    img_sum = np.array(img_sum)
    print(img_sum.shape)
    assert np.max(img_sum) == np.max(mask_sum)
    assert np.min(img_sum) == np.min(mask_sum)
    eval_mat = calculate_eval_matrix(2, mask_sum, img_sum)
    print('indi IoU:', calculate_IoU(eval_mat))
    print('IoU:', np.mean(calculate_IoU(eval_mat)))
    print('acc:', calculate_acc(eval_mat))
    single_metric(img_sum, mask_sum, sufix=sufix)


def post_processing(image):
    image = np.squeeze(image)
    contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('No breast')
        return image
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    temp = np.zeros_like(image)

    thresh = cv2.fillPoly(temp, [contours], (255, 255, 255))
    # plt.figure()
    # plt.imshow(thresh * 255, cmap='gray')
    #
    # plt.show()
    return thresh


def single_metric(preds, masks, sufix):
    sufix = sufix[:-5]
    iou, acc = [], []
    for pred, mask in zip(preds, masks):
        eval_mat = calculate_eval_matrix(2, mask, pred)
        # print(calculate_IoU(eval_mat))
        iou.append(np.mean(calculate_IoU(eval_mat)))
        acc.append(calculate_acc(eval_mat))
    std_iou = np.std(iou, ddof=1)
    std_acc = np.std(acc, ddof=1)
    var_iou = np.var(iou)
    var_acc = np.var(acc)
    data = pd.DataFrame({'iou': iou, 'acc': acc})
    dataframe = pd.DataFrame({'std_iou': std_iou.tolist(), 'std_acc': std_acc.tolist(), 'var_iou': var_iou.tolist(),
                              'var_acc': var_acc.tolist(), }, index=[0])
    dataframe = pd.concat([data, dataframe])
    dataframe.to_csv(f"{sufix}.csv", index=True, sep=',')
    print(std_acc, std_iou, var_acc, var_iou)
    print('-' * 20)


class infer_gui():
    def __init__(self, models, size=[64, 128, 256, 512]):
        # self.model = unet_train.load_from_checkpoint(models)
        self.model_CKPT = torch.load(models)
        self.model = UNET(in_channels=3, out_channels=1, features=size).half().cuda()
        # self.model = UNET_S(in_channels=3, out_channels=1).half().cuda()

        loaded_dict = self.model_CKPT['state_dict']
        prefix = 'model.'
        n_clip = len(prefix)
        adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                        if k.startswith(prefix)}
        self.model.load_state_dict(adapted_dict)
        IMAGE_HEIGHT = 274  # 1096 originally  0.25
        IMAGE_WIDTH = 484  # 1936 originally
        self.infer_xform = A.Compose(
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

        self.infer_xform2 = A.Compose(
            [
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
            ],
        )

    def forward(self, image):
        input = np.array(Image.open(image), dtype=np.uint8)
        resize_xform = A.Compose(
            [
                A.Resize(height=input.shape[0], width=input.shape[1]),

                ToTensorV2(),
            ],
        )
        input = self.infer_xform(image=input)
        x = input["image"].cuda()

        x = torch.unsqueeze(x, dim=0)

        y_hat = self.model(x)
        preds = torch.sigmoid(y_hat.squeeze())
        preds = (preds > 0.6).float()

        preds = resize_xform(image=preds.cpu().numpy())
        preds = preds["image"].squeeze(0).numpy()

        return preds


if __name__ == "__main__":
    '''
    until 31.08.2021 best model is :: .\goodmodel\epoch=71-valid_IOU=0.821612.ckpt
    
    
    '''
    modelslist = []
    for root, dirs, files in os.walk(r".\goodmodel"):
        for file in files:
            if file.endswith('.ckpt'):
                modelslist.append(os.path.join(root, file))
    for idx, model in enumerate(modelslist):
        print(f'{idx}:', model)
    picked = int(input('Model:'))
    assert abs(picked) <= len(modelslist) - 1
    print('inference model : ', modelslist[picked])
    sufix = modelslist[picked].split('\\')[-1]
    # g=infer_gui(modelslist[0])
    # image=g.forward(r'C:\Users\z00461wk\Desktop\Pressure_measure_activate tf1x\Camera_util/breast.jpg')
    # print(image.shape)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    # infer(modelslist[picked], './data/val_images',sufix=sufix)
    infer(modelslist[picked], './testdata',sufix=sufix)
    # infer(modelslist[picked], r'F:\semantic_segmentation_unet\Cam62-71\20181215-06.00', sufix=sufix)
    # metrics(modelslist[picked], './data/val_images', './data/val_masks',sufix=sufix)
