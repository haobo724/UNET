import glob
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd

from caculate import calculate_eval_matrix, calculate_IoU, calculate_acc
from utils import cal_std_mean

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image as Image
from train import mutil_train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

PIN_MEMORY = True
TEST_DIR = 'data/test_set'
TEST_MASK_DIR = 'data/test_set_mask'


# TRAIN_IMG_DIR = "clinic_old/"


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--mode_size', type=int, default=32)
    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default='Unet')

    return parser


def metrics(models, img_dir, mask_dir, sufix='sufix', post=True):
    IMAGE_HEIGHT = 256  # 1096 originally  0.25
    IMAGE_WIDTH = 448  # 1936 originally
    TRAIN_IMG_DIR = "data/clinic/"

    if 'mixed' in models:
        TRAIN_IMG_DIR = "data/mixed_set/"

    elif 'elbows' in models:
        TRAIN_IMG_DIR = "data/elbows/"
        # mask_dir = 'data/elbows_mask'
        # img_dir = 'data/elbows'
    elif 'eighth' in models:
        IMAGE_HEIGHT = 274 // 8  # 1096 originally  0.25
        IMAGE_WIDTH = 484 // 8  # 1936 originally
    temp_flag = True
    if temp_flag:
        mean_value, std_value = cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        mean_value, std_value = [0, 0, 0], [1, 1, 1]
    if img_dir is None or models is None:
        ValueError('raw_dir or model is missing')
    filename_mask = sorted(glob.glob(os.path.join(mask_dir, "*.tiff")))
    filename_img = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    mask_sum = []
    img_sum = []
    for mask in filename_mask:
        mask_img = cv2.imread(mask)[..., 0]
        mask_sum.append(mask_img)
    mask_sum = np.array(mask_sum)

    parser = ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    model = mutil_train.load_from_checkpoint(models, hparams=vars(args))
    infer_xform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=mean_value,
                std=std_value,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    with torch.no_grad():
        for index in range(len(filename_img)):
            input = np.array(Image.open(filename_img[index]), dtype=np.uint8)
            resize_xform = A.Compose(
                [
                    A.Resize(height=input.shape[0], width=input.shape[1], interpolation=cv2.INTER_NEAREST),

                    ToTensorV2(),
                ],
            )
            input = infer_xform(image=input)
            x = input["image"].cuda()

            x = torch.unsqueeze(x, dim=0)
            model.eval()
            y_hat = model(x)
            preds = torch.softmax(y_hat, dim=1)

            preds = preds.argmax(dim=1).float()
            preds = preds.squeeze()

            preds = resize_xform(image=preds.cpu().numpy())
            preds = preds["image"].numpy() * 1
            # plt.figure()
            # plt.imshow(preds[0])
            # plt.show()
            if post:
                preds = post_processing(preds) / 255
            preds = preds.squeeze(0)
            img_sum.append(preds)
    img_sum = np.array(img_sum)
    # assert np.max(img_sum) == np.max(mask_sum)
    assert np.min(img_sum) == np.min(mask_sum)

    eval_mat = calculate_eval_matrix(len(np.unique(mask_sum)), mask_sum, img_sum)
    print('indi IoU:', calculate_IoU(eval_mat))
    print('IoU:', np.mean(calculate_IoU(eval_mat)[1:]))
    print('-' * 20)

    # print('acc:', calculate_acc(eval_mat))
    std_acc, std_iou, var_acc, var_iou = single_metric(img_sum, mask_sum, sufix=sufix, post=post)
    return np.mean(calculate_IoU(eval_mat)), calculate_acc(eval_mat), std_acc, std_iou, var_acc, var_iou


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


def single_metric(preds, masks, sufix, post=True):
    sufix = sufix[:-5]
    if post:
        sufix = 'post' + sufix
    else:
        sufix = 'no-post' + sufix

    iou, acc = [], []
    for pred, mask in zip(preds, masks):
        eval_mat = calculate_eval_matrix(2, mask, pred)
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
    # print(std_acc, std_iou, var_acc, var_iou)
    print('-' * 20)
    return std_acc, std_iou, var_acc, var_iou


if __name__ == "__main__":
    '''
    until 31.08.2021 best model is :: .\goodmodel\epoch=71-valid_IOU=0.821612.ckpt


    '''
    modelslist = []
    for root, dirs, files in os.walk(r"model_pixel"):
        for file in files:
            if file.endswith('.ckpt'):
                modelslist.append(os.path.join(root, file))
    for idx, model in enumerate(modelslist):
        print(f'{idx}:', model)

    iou = []
    acc = []
    std_accs = []
    std_ious = []
    var_accs = []
    var_ious = []
    for i in range(len(modelslist)):
        # metrics(modelslist[picked], './data/val_images', './data/val_masks',sufix=sufix)
        sufix = modelslist[i].split('\\')[-1]
        print(modelslist[i], ':')
        i, a, std_acc, std_iou, var_acc, var_iou = metrics(modelslist[i], TEST_DIR, TEST_MASK_DIR,
                                                           sufix=sufix, post=False)
        # iou.append(i)
        # acc.append(a)
        # std_accs.append(std_acc)
        # std_ious.append(std_iou)
        # var_accs.append(var_acc)
        # var_ious.append(var_iou)
    # print('iou', np.array(iou).mean())
    # print('acc', np.array(acc).mean())
    # print('std_accs', np.array(std_accs).mean())
    # print('std_ious', np.array(std_ious).mean())
    # print('var_accs', np.array(var_accs).mean())
    # print('var_ious', np.array(var_ious).mean())
