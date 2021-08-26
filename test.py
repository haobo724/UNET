import cv2
import numpy as np
import glob, os
import tqdm
import albumentations as A

mask_dir = r'C:\ChangLiu\MasterThesis\TrainSet\full_13012020\Label_class_1'
img_dir = r'C:\ChangLiu\MasterThesis\TrainSet\full_13012020\Original_img\00001.jpg'
save_dir = './test_mask/'
IMG = cv2.imread(img_dir)
print(IMG.shape)
my_transform = A.Compose([A.Resize(512, 512),
                          A.ColorJitter(brightness=0.5, saturation=0.3, contrast=0.5, hue=0.1, p=1),
                          # A.ColorJitter(saturation=2,hue=0),
                          ])
while True:
    new_img = my_transform(image=IMG)['image']
    cv2.imshow('new', new_img)
    cv2.waitKey()


# filename = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
# filename_mask = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
# for img ,mask in tqdm.tqdm(zip(filename,filename_mask)):
#     prefix=img.split('\\')[-1]
#     # prefix=prefix.split('.')[0]
#     img_np=cv2.imread(img)
#     mask_np=cv2.imread(mask,0)/255
#     mask_np[mask_np==0]=0.5
#
#     result=np.stack(((img_np[...,channel]*mask_np).astype('uint8') for channel in range(3)),axis=-1)
#     # result=img_np[mask_np].reshape((w,h))
#
#     # result=np.dstack((result,result,result))
#     # cv2.namedWindow('temp')
#     # cv2.resizeWindow('temp',1000,600)
#     # cv2.imshow('temp',result)
#     # cv2.waitKey()
#
#     cv2.imwrite(save_dir+prefix,result)
