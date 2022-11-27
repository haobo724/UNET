import glob
import os
import time
from argparse import ArgumentParser

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from caculate import calculate_eval_matrix, calculate_IoU, calculate_acc
from infer_model import mapping_color
from mutil_train import mutil_train
from utils import cal_std_mean

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
IMAGE_HEIGHT = 480  # 1096 originally  0.25
IMAGE_WIDTH = 640  # 1936 originally
PIN_MEMORY = True

# TEST_DIR = r'F:\Siemens\GreenPointpick\test_input'
TEST_DIR = r'F:\semantic_segmentation_unet\data\test_new'
# TEST_MASK_DIR = r'F:\Siemens\GreenPointpick\test_mask'
TEST_MASK_DIR = r'F:\semantic_segmentation_unet\data\test_new_mask'


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--mode_size', type=int, default=32)
    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default='Unet')
    parser.add_argument('--lr', type=float, default=1e-4)

    return parser


def metrics(models, img_dir, mask_dir, mean_value, std_value, sufix='', matrix_dir='./test_pers_matrix', roi=False,
            only_breast=True, saved=False):
    if img_dir is None or models is None:
        ValueError('raw_dir or model is missing')

    filename_mask = sorted(glob.glob(os.path.join(mask_dir, "*.tiff")))
    filename_img = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_sum = []
    pred_sum = []
    assert len(filename_mask) == len(filename_img)

    if roi:
        perspektiv_matrix = sorted(glob.glob(os.path.join(matrix_dir, "*.npy")))
        for mask, M in zip(filename_mask, perspektiv_matrix):
            mask_img = cv2.imread(mask)[..., 0].astype(np.uint8)
            if only_breast:
                mask_img = np.where(mask_img > 0, 1, 0).astype(np.uint8)

            mask_img = post_processing_roi(mask_img, M)
            mask_sum.append(mask_img)


    else:
        for mask in filename_mask:
            mask_img = cv2.imread(mask)[..., 0].astype(np.uint8)
            if only_breast:
                mask_img = np.where(mask_img > 0, 1, 0).astype(np.uint8)

            mask_sum.append(mask_img)
    parser = ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()
    args.model = os.path.basename(models).split('_')[0]
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

    input_sum = []
    time_consume = []
    with torch.no_grad():
        for index in range(len(filename_img)):
            input = cv2.imread(filename_img[index]).astype(np.uint8)
            input_height, input_width = input.shape[0], input.shape[1]
            if roi:
                input_sum.append(post_processing_roi(input, perspektiv_matrix[index]))
            else:
                input_sum.append(input)
            resize_xform = A.Compose(
                [
                    A.Resize(height=input_height, width=input_width, interpolation=cv2.INTER_NEAREST),

                ],
            )

            input = infer_xform(image=input)
            start_time = time.time()

            x = input["image"].cuda()

            x = torch.unsqueeze(x, dim=0)
            model.eval()
            y_hat = model(x)
            preds = torch.softmax(y_hat, dim=1)
            preds = preds.argmax(dim=1).float()
            preds = preds.squeeze()

            preds = resize_xform(image=preds.cpu().numpy())
            preds = preds["image"] * 1
            if only_breast:
                preds = np.where(preds > 0, 1, 0).astype(np.uint8)
                # preds = post_processing(preds) / 255
            if roi:
                preds = post_processing_roi(preds, perspektiv_matrix[index])
                preds = np.around(preds)


            end_time = time.time() - start_time
            time_consume.append(end_time)
            pred_sum.append(preds)
    time_consume = np.mean(time_consume)
    print('time used:', time_consume)
    iou = 0
    counter = 0
    acc = 0
    for mask, pred, img in zip(mask_sum, pred_sum, input_sum):

        if only_breast:
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)
            eval_mat = calculate_eval_matrix(2, mask, pred)
            iou += calculate_IoU(eval_mat)
            acc += calculate_acc(eval_mat)
        else:
            eval_mat = calculate_eval_matrix(3, mask, pred)
            iou += calculate_IoU(eval_mat)
            acc += calculate_acc(eval_mat)

        # img = cv2.resize(img, ( input_width,input_height))
        pred_color = np.stack([pred for _ in range(3)], axis=-1)
        mask_color = np.stack([mask for _ in range(3)], axis=-1)
        pc = mapping_color(pred_color)
        mc = mapping_color(mask_color)
        input_pc = cv2.addWeighted(img.astype(np.uint8), 0.8, pc.astype(np.uint8), 0.2, 1)
        input_mc = cv2.addWeighted(img.astype(np.uint8), 0.8, mc.astype(np.uint8), 0.2, 1)

        # cv2.imwrite(rf'./MA_infer/{counter}.jpg', img)
        # cv2.imwrite(rf'./MA_infer/{counter}_mask.jpg', input_mc)
        # cv2.imwrite(rf'./MA_infer/{counter}_pred.jpg', input_pc)
        counter += 1
        if saved:

            # input_pc = cv2.resize(input_pc, (640, 480))
            # img = cv2.resize(img, (640, 480))
            # input_mc = cv2.resize(input_mc, (640, 480))
            if roi:

                cv2.imwrite(rf".\MA_infer_result\roi_{args.model}_Result_{counter}.png", input_pc)
                cv2.imwrite(rf".\MA_infer_result\roi_reference_{counter}.png", img)
                cv2.imwrite(rf".\MA_infer_result\roi_reference_mask_{counter}.png", input_mc)
            else:
                cv2.imwrite(rf".\MA_infer_result\{args.model}_Result_{counter}.png", input_pc)
                cv2.imwrite(rf".\MA_infer_result\reference_{counter}.png", img)
                cv2.imwrite(rf".\MA_infer_result\reference_mask_{counter}.png", input_mc)
    if saved:
        return

        # print(iou)
    # print('Only brest iou:', iou/len(pred_sum))
    mIou = np.mean(iou / len(pred_sum))
    acc = np.mean(acc / len(pred_sum))

    print('indi IoU:', iou / len(pred_sum))
    if only_breast:
        print('bIoU:', mIou)
    else:
        print('mIou:', mIou)
    print('ACC:', acc)
    print('-' * 20)

    std_acc, std_iou, var_acc, var_iou = single_metric(pred_sum, mask_sum, sufix=sufix, roi=roi,
                                                       only_breast=only_breast)
    return mIou, acc, std_acc, std_iou, var_acc, var_iou, time_consume


def post_processing_roi(image, npy_path):
    M = np.load(npy_path)
    image = cv2.warpPerspective(image, M, (640, 480))
    return image


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


def single_metric(preds, masks, sufix, roi=False, only_breast=False):
    sufix = sufix[:-5]
    if roi:
        sufix = 'post_' + sufix
    else:
        sufix = 'no-post_' + sufix

    iou, acc = [], []
    for pred, mask in zip(preds, masks):
        if only_breast:
            eval_mat = calculate_eval_matrix(2, mask, pred)
        else:
            eval_mat = calculate_eval_matrix(3, mask, pred)

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
    if roi:
        dataframe.to_csv(f"MA_CSV/{sufix}.csv", index=True, sep=',')
    else:
        dataframe.to_csv(f"MA_CSV/{sufix}_roi.csv", index=True, sep=',')

    # print(std_acc, std_iou, var_acc, var_iou)
    print('-' * 20)
    return std_acc, std_iou, var_acc, var_iou


def start_metrics(model_name='', model_index=None, show=False):
    modelslist = []
    for root, dirs, files in os.walk(r"MA_model"):
        for file in files:
            if file.startswith(model_name):
                modelslist.append(os.path.join(root, file))
    TRAIN_IMG_DIR = r"F:\semantic_segmentation_unet\data\All_clinic"

    mean_value, std_value = cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)

    iou = []
    acc = []
    std_accs = []
    std_ious = []
    var_accs = []
    var_ious = []
    time_used = []
    counter = 0
    best_iou = 0
    if model_index is not None:
        sufix = os.path.basename(modelslist[model_index])
        if show:
            Final_TEST_DIR = '.\MA_TEST_data'
            Final_TEST_MASK_DIR = 'F:\semantic_segmentation_unet\MA_TEST_mask'
            metrics(modelslist[model_index], Final_TEST_DIR,
                    Final_TEST_MASK_DIR,
                    mean_value, std_value,
                    sufix=sufix, matrix_dir='F:\semantic_segmentation_unet\MA_TEST_npy',roi=False,
                    only_breast=False,saved=False)
        else:
            miou, a, std_acc, std_iou, var_acc, var_iou, time_consume = metrics(modelslist[model_index], TEST_DIR,
                                                                            TEST_MASK_DIR,
                                                                            mean_value, std_value,
                                                                            sufix=sufix, roi=True,
                                                                            only_breast=False,saved=False)

            print('std_acc', np.array(std_acc).mean())

        return
    for i in range(len(modelslist)):
        sufix = os.path.basename(modelslist[i])
        print(modelslist[i], ':')
        miou, a, std_acc, std_iou, var_acc, var_iou, time_consume = metrics(modelslist[i], TEST_DIR, TEST_MASK_DIR,
                                                                            mean_value, std_value,
                                                                            sufix=sufix, roi=False,
                                                                            only_breast=True)
        if miou > best_iou:
            best_iou = miou
            counter = i

        iou.append(miou)
        acc.append(a)
        std_accs.append(std_acc)
        std_ious.append(std_iou)
        var_accs.append(var_acc)
        var_ious.append(var_iou)
        time_used.append(time_consume)

    print('std_acc', np.array(std_acc).mean())
    # print('iou', np.array(iou).mean())
    # print('acc', np.array(acc).mean())
    # print('time_used', np.array(time_used[1:]).mean())
    # sufix = os.path.basename(modelslist[counter])
    # Final_TEST_DIR = '.\MA_TEST_data'
    # Final_TEST_MASK_DIR = 'F:\semantic_segmentation_unet\MA_TEST_mask'
    # print(f'best model is {counter}')
    # metrics(modelslist[counter], Final_TEST_DIR, Final_TEST_MASK_DIR,
    #         mean_value, std_value,
    #         sufix=sufix, matrix_dir='F:\semantic_segmentation_unet\MA_TEST_npy', roi=True,
    #         only_breast=False,
    #         show=True)


if __name__ == "__main__":
    # model_name = ['unetplusplus-resnet34', 'Unet-vgg16', 'Unet-at-res', 'Unet-res']
    model_name = ['Unet-res']
    # best unet res is 4
    for m in model_name:
        start_metrics(m,3)
