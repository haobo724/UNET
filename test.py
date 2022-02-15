import os
from sklearn import model_selection
import os

from sklearn import model_selection

mask_dir = r'C:\ChangLiu\MasterThesis\TrainSet\full_13012020\Label_class_1'
img_dir = r'C:\Users\z00461wk\Desktop\haobo\semantic_segmentation_unet\data\train_images'
save_dir = './test_mask/'

X = glob.glob('./data/all_images/*.jpg')
y = glob.glob('./data/all_masks/*.jpg')

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 1234)
img_name=[]
for i in X_train:
    i = os.path.split(i)[-1]
    img_name.append(i)

# img = os.listdir(img_dir)
# for i in img:
#     im = cv2.imread(os.path.join(img_dir,i))
#     mean1=np.mean(im[...,:0])
#     mean2=np.mean(im[...,:1])
#     mean3=np.mean(im[...,:2])
# print(mean1)
# IMG = cv2.imread(img_dir)
# print(IMG.shape)
# my_transform = A.Compose([A.Resize(512, 512),
#                           A.ColorJitter(brightness=0.5, saturation=0.3, contrast=0.5, hue=0.1, p=1),
#                           # A.ColorJitter(saturation=2,hue=0),
#                           ])
# while True:
#     new_img = my_transform(image=IMG)['image']
#     cv2.imshow('new', new_img)
#     cv2.waitKey()


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
