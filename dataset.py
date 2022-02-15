import glob
import os

import numpy as np
from PIL import Image
from sklearn import model_selection
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None,val=False,seed=1234):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # self.images = os.listdir(image_dir)
        # self.masks = os.listdir(mask_dir)
        self.images = []
        self.masks = []
        X = glob.glob('./data/all_images/*.jpg')
        y = glob.glob('./data/all_masks/*.jpg')

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)
        if val:
            for i,j in zip(X_test,y_test):
                i = os.path.split(i)[-1]
                j = os.path.split(j)[-1]
                self.images.append(i)
                self.masks.append(j)
        else:
            for i,j in zip(X_train,y_train):
                i = os.path.split(i)[-1]
                j = os.path.split(j)[-1]
                self.images .append(i)
                self.masks .append(j)
        print(len(self.images))
        print(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        if np.max(mask)!=1:
            if len(np.unique(mask))==2:
                low, high = np.unique(mask)
                mask[mask == high] = 1.0
                mask[mask == low] = 0.0
            else:
                mask[mask > 0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

