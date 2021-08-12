import os
import time

import cv2
import numpy as np

import glob
from caculate import calculate_eval_matrix,calculate_IoU,calculate_acc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image as Image
import torchvision
from train import unet_train
import matplotlib.pyplot  as plt
from model import UNET,UNET_S

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 8
IMAGE_HEIGHT = 274  # 1096 originally  0.25
IMAGE_WIDTH = 484  # 1936 originally
PIN_MEMORY = True
LOAD_MODEL = False


def infer(models, raw_dir):
    if raw_dir is None or models is None:
        ValueError('raw_dir or model is missing')

    # train_imgs = [
    #     {keys[0]: img, keys[1]: seg} for img, seg in
    #     zip(images[350:360], labels[350:360])
    # ]
    # val_imgs = [
    #     {keys[0]: img, keys[1]: seg} for img, seg in
    #     zip(images[350:360], labels[350:360])
    # ]
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
    print(infer_data)
    filename = os.listdir(raw_dir)

    folder = "saved_images/"
    curtime = time.time()
    model = unet_train.load_from_checkpoint(models)

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
        model.eval()
        y_hat = model(x)
        preds = torch.sigmoid(y_hat.squeeze())
        preds = (preds > 0.6).float()

        preds = resize_xform(image=preds.cpu().numpy())
        preds = preds["image"].numpy()
        timeend = time.time()

        preds = np.vstack((preds, preds, preds))
        preds = np.expand_dims(preds, axis=0)

        input_copy = infer_xform2(image=input_copy)["image"]
        input_copy = np.transpose(input_copy, (2, 0, 1))
        input_copy = np.expand_dims(input_copy, axis=0)

        saved = np.vstack((preds, input_copy))
        saved = torch.tensor(saved)

        torchvision.utils.save_image(
            saved, f"{folder}/infer_{filename[index]}"
        )
        print(f'At {index} image used:{timeend - timebegin} s')

    end = time.time()
    print(f'Totally used:{end - curtime} s')

def metrics(models,img_dir,mask_dir):
    if img_dir is None or models is None:
        ValueError('raw_dir or model is missing')
    filename = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
    filename_img = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_sum=[]
    img_sum=[]
    for mask in filename:
        mask_img=cv2.imread(mask,0)/255
        mask_sum.append(mask_img)
    mask_sum=np.array(mask_sum)
    print(mask_sum.shape)
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
        preds = preds["image"].numpy()*1
        preds=preds.squeeze()
        img_sum.append(preds)
    print(time.time()-curtime)
    img_sum=np.array(img_sum)
    print(img_sum.shape)
    assert np.max(img_sum)==np.max(mask_sum)
    assert np.min(img_sum)==np.min(mask_sum)
    eval_mat=calculate_eval_matrix(2,mask_sum,img_sum)
    print(calculate_IoU(eval_mat))
    print(calculate_acc(eval_mat))
class infer_gui():
    def __init__(self, models):
        # self.model = unet_train.load_from_checkpoint(models)
        self.model_CKPT = torch.load(models)
        self.model = UNET(in_channels=3, out_channels=1).half().cuda()
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
    modelslist = []
    for root, dirs, files in os.walk(r".\goodmodel"):
        for file in files:
            if file.endswith('.ckpt'):
                modelslist.append(os.path.join(root, file))
    print(modelslist)
    print(modelslist[-1])
    # g=infer_gui(modelslist[0])
    # image=g.forward(r'C:\Users\z00461wk\Desktop\Pressure_measure_activate tf1x\Camera_util/breast.jpg')
    # print(image.shape)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    # infer(modelslist[-1], './testdata')
    metrics(modelslist[-1], './data/val_images','./data/val_masks')
