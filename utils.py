import os
from argparse import ArgumentParser

import torch
import torchvision
from setuptools import glob
from sklearn import model_selection
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CarvanaDataset, CarvanaDataset_multi, LeafData
import albumentations as A
from albumentations.pytorch import ToTensorV2


def cal_std_mean(TRAIN_IMG_DIR,IMAGE_HEIGHT,IMAGE_WIDTH):
    augs = A.Compose([A.Resize(height = IMAGE_HEIGHT,
                           width  = IMAGE_WIDTH),
        A.Normalize(mean=(0, 0, 0),
                    std=(1, 1, 1)),
        ToTensorV2()])
    imgs = glob.glob(TRAIN_IMG_DIR + '*.jpg')
    print(imgs)
    image_dataset = LeafData(data=imgs,
                             transform=augs)
    image_loader = DataLoader(image_dataset,
                              batch_size=4,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 2, 3])

        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])
    # pixel count
    count = len(glob.glob(TRAIN_IMG_DIR + '/*.jpg')) * IMAGE_HEIGHT*IMAGE_WIDTH
    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    torch.cuda.empty_cache()

  #mean: tensor([0.2292, 0.2355, 0.3064])
  #std:  tensor([0.2448, 0.2450, 0.2833])

    # output
    print('mean: ' + str(tuple(total_mean)))
    print('std:  ' + str(tuple(total_std)))
    return total_mean,total_std

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_classes(path):
    # Loads class labels at 'path'
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def parse_data_cfg(path):
    """Parses the data configuration file"""
    print('data_cfg ： ', path)
    options = dict()
    # options['gpus'] = '0,1,2,3'
    # options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=True,
        seed=1234
):
    X = glob.glob('./data/all_images/*.jpg')
    y = glob.glob('./data/all_masks/*.jpg')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)
    train_img = []
    train_mask = []
    val_img = []
    val_mask = []

    for i, j in zip(X_test, y_test):
        i = os.path.split(i)[-1]
        j = os.path.split(j)[-1]
        val_img.append(i)
        val_mask.append(j)
    for i, j in zip(X_train, y_train):
        i = os.path.split(i)[-1]
        j = os.path.split(j)[-1]
        train_img.append(i)
        train_mask.append(j)
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        imgs=train_img,
        masks=train_mask

    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        imgs=val_img,
        masks=val_mask

    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,

    )

    return train_loader, val_loader


def get_loaders_multi(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=True,
        seed=1234
):
    X = glob.glob(train_dir + '/*.jpg')
    y = glob.glob(train_maskdir + '/*.tiff')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)
    train_img = []
    train_mask = []
    val_img = []
    val_mask = []
    for i, j in zip(X_test, y_test):
        i = os.path.basename(i)
        j = os.path.basename(j)
        val_img.append(i)
        val_mask.append(j)
    for i, j in zip(X_train, y_train):
        i = os.path.basename(i)
        j = os.path.basename(j)
        train_img.append(i)
        train_mask.append(j)

    train_ds = CarvanaDataset_multi(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        imgs=train_img,
        masks=train_mask

    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset_multi(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        imgs=val_img,
        masks=val_mask

    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,

    )

    return train_loader, val_loader


def get_testloaders(test_dir,
                    test_maskdir,
                    batch_size,
                    test_transform,
                    num_workers,
                    pin_memory=True, ):
    X = glob.glob(r'testdata\*.jpg')
    X = X+glob.glob(r'testdata\*.PNG')
    y = glob.glob(r'testdata\*.jpg')
    val_img = []
    val_mask = []
    for i, j in zip(X, y):
        i = os.path.split(i)[-1]
        j = os.path.split(j)[-1]
        val_img.append(i)
        val_mask.append(j)
    test_ds = CarvanaDataset_multi(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
        imgs=val_img,
        masks=val_mask,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return test_loader


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--mode_size', type=int, default=64)
    parser.add_argument("--model", type=str, default='Unet')
    parser.add_argument("--Continue", type=bool, default=False)

    return parser



if __name__ == "__main__":
    TRAIN_IMG_DIR = "data/clinic/"

    cal_std_mean(TRAIN_IMG_DIR)
