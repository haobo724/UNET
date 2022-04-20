import cv2
import numpy as np
mask = cv2.imread(r'F:\semantic_segmentation_unet\data\clinic_mask\00287-labels.tiff')[...,0].astype(np.float32)

print(mask.shape)
# print(dtype(test_img))
print(np.unique(mask))