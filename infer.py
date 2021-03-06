import glob
import os
import shutil
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from caculate import calculate_eval_matrix, calculate_IoU, calculate_acc
from utils import cal_std_mean

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image as Image
import torchvision
from mutil_train import unet_train
from model import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 8
IMAGE_HEIGHT = 480  # 1096 originally  0.25
IMAGE_WIDTH = 640  # 1936 originally
PIN_MEMORY = True
LOAD_MODEL = False
TEST_DIR = 'data/test_set'


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--mode_size', type=int, default=32)
    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default='Unet')

    return parser


def infer(models, raw_dir, sufix):
    if raw_dir is None or models is None:
        ValueError('raw_dir or model is missing')
    mean_value = (0.3651, 0.3123, 0.2926)
    std_value = (0.3383, 0.3004, 0.2771)
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

    infer_xform2 = A.Compose(
        [
            A.Normalize(

                max_pixel_value=255.0,
            ),
        ],
    )

    infer_data = sorted(glob.glob(os.path.join(raw_dir, "*.PNG")))
    infer_data2 = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    infer_data += infer_data2

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
            # input = np.array(Image.open(infer_data[index]), dtype=np.uint8)[..., 0:3]
            input = cv2.imread(infer_data[index])
            input_copy= input
            # plt.imshow(input_copy)
            # plt.show()
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
            # post_pred = np.expand_dims(post_pred, axis=0)
            print(post_pred.shape)

            # input_copy = infer_xform2(image=input_copy)["image"]
            post_pred = np.transpose(post_pred, (1, 2, 0))
            print(post_pred.shape)

            # input_copy = np.expand_dims(input_copy, axis=0)
            print(input_copy.shape)

            saved = np.concatenate((input_copy, post_pred),axis=0)
            # saved = torch.tensor(saved)
            filename = infer_data[index].split('\\')[-1]
            print(filename)
            cv2.imwrite(f"{folder}/infer_{filename}",saved)
            # torchvision.utils.save_image(
            #     saved,
            # )
            print(f'At {index} image used:{timeend - timebegin} s')

    end = time.time()
    print(f'Totally used:{end - curtime} s')


def metrics(models, img_dir, mask_dir, sufix='sufix', post=True):
    mean_value =(0.3651, 0.3123, 0.2926)
    std_value=(0.3383, 0.3004, 0.2771)
    #
    # mean_value =(0.0, 0.0, 0.0)
    # std_value=(0.0, 0.0, 0.0)
    if img_dir is None or models is None:
        ValueError('raw_dir or model is missing')
    filename_mask = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
    filename_img = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    # X = glob.glob('./data/all_images/*.jpg')
    # y = glob.glob('./data/all_masks/*.jpg')
    # seed = models.split('_')[-1][:4]
    # print('seed:', seed)
    # _, filename_img, _, filename_mask = model_selection.train_test_split(X, y, test_size=0.25, random_state=int(seed))

    mask_sum = []
    img_sum = []
    for mask in filename_mask:
        mask_img = cv2.imread(mask, 0)/255
        mask_sum.append(mask_img)
    mask_sum = np.array(mask_sum)
    # test=torch.load(models)
    # mode_size=test['state_dict']['model.ups.0.weight'].size()[0]/16
    # models['hyper_parameters'][0].update({"mode_size": int(mode_size)})
    parser = ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    model = unet_train.load_from_checkpoint(models, hparams=vars(args))
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
            input = cv2.imread(filename_img[index])
            # input = cv2.cvtColor(input,cv2.COLOR_BGR2RGB)
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
            preds = torch.sigmoid(y_hat.squeeze())
            preds = (preds > 0.6).float()

            preds = resize_xform(image=preds.cpu().numpy())
            preds = preds["image"].numpy() * 1

            if post:
                preds = post_processing(preds) / 255
            preds = preds.squeeze()
            img_sum.append(preds)
    img_sum = np.array(img_sum)

    assert np.max(img_sum) == np.max(mask_sum)
    assert np.min(img_sum) == np.min(mask_sum)
    eval_mat = calculate_eval_matrix(2, mask_sum, img_sum)
    print('indi IoU:', calculate_IoU(eval_mat))
    print('IoU:', np.mean(calculate_IoU(eval_mat)))
    print('acc:', calculate_acc(eval_mat))
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
    return std_acc, std_iou, var_acc, var_iou




if __name__ == "__main__":
    '''
    until 31.08.2021 best model is :: .\goodmodel\epoch=71-valid_IOU=0.821612.ckpt
    
    
    '''
    modelslist = []
    for root, dirs, files in os.walk(r"F:\semantic_segmentation_unet\model_crosnew"):
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

        # i, a, std_acc, std_iou, var_acc, var_iou = metrics(modelslist[i], './test_images', './test_maskes',
        #                                                    sufix=sufix, post=True)

        infer(modelslist[0],'./test_images',sufix='124')
        break
        iou.append(i)
        # acc.append(a)
        # std_accs.append(std_acc)
        # std_ious.append(std_iou)
        # var_accs.append(var_acc)
        # var_ious.append(var_iou)
    print('iou', np.array(iou).mean())
    print('acc', np.array(acc).mean())
    print('std_accs', np.array(std_accs).mean())
    print('std_ious', np.array(std_ious).mean())
    print('var_accs', np.array(var_accs).mean())
    print('var_ious', np.array(var_ious).mean())
