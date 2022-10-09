import glob
import os

import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold

TRAIN_IMG_DIR = "F:\semantic_segmentation_unet\data\All_clinic"
TRAIN_MASK_DIR = "F:\semantic_segmentation_unet\data\All_clinic_mask"
if __name__ == "__main__":

    images = os.listdir(TRAIN_IMG_DIR)
    masks = os.listdir(TRAIN_MASK_DIR)
    images = np.array(images)
    masks = np.array(masks)
    kf = KFold(n_splits=5,shuffle=True,random_state=1002)
    i =1
    for train, val in kf.split(images):
        idx_train = np.array(train).astype(np.int_)
        idx_val= np.array(val).astype(np.int_)
        images_train =images[idx_train]
        images_val =images[idx_val]
        masks_train =masks[idx_train]
        masks_val =masks[idx_val]
        np.save(f'Nr_{i}',[images_train,images_val,masks_train,masks_val])
        # print(X_train)
        # print("%s %s" % (train, test))
        file = np.load(f'Nr_{i}.npy',allow_pickle=True)
        print(file[0])
        print(file[1])
        print(file[2].shape)
        print(file[3].shape)
        i+=1
