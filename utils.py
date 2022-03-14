import os

import torch
import torchvision
from setuptools import glob
from sklearn import model_selection
from torch.utils.data import DataLoader

from dataset import CarvanaDataset,CarvanaDataset_multi


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
    print('data_cfg ： ',path)
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
    X = glob.glob('./data/train_set/*.jpg')
    y = glob.glob('./data/train_set_mask/*.tiff')
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

    X = glob.glob(r'C:\Users\94836\Desktop\test_data/*.jpg')
    y = glob.glob(r'C:\Users\94836\Desktop\test_data/*.jpg')
    val_img=[]
    val_mask=[]
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


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            # print("pred:",preds.shape,"yshapenot:",y.shape,"yshape:",y.unsqueeze(1).shape)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
