import os
from sklearn import model_selection
import os,numpy as np
import torch
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss,BCELoss
from sklearn import model_selection
import monai
loss = CrossEntropyLoss(reduction='sum')
BCE_LOG_LOSS = BCEWithLogitsLoss()
# target =np.random.randn(3,3)
# target =torch.randn((4,10,10)).long()
target =torch.rand((1,10,12)).random_(0,2).long()
target_onhot = torch.nn.functional.one_hot(target).float()
target_onhot = torch.moveaxis(target_onhot,-1,1)
print(target_onhot.size())

input = torch.ones((1,2,10,12))

input_soft =torch.softmax(input,dim=1)
input_log =torch.log(input_soft)
res=torch.nn.functional.nll_loss(input_log, target)

result_manuel = -torch.sum(target_onhot*input_log)/target.shape[0]
result = loss.forward(input,target_onhot)
result_normal = loss.forward(input,target)
# result_normal2 = loss.forward(input,target.unsqueeze(1))
print(result,result_normal,result_manuel)
input_onechannel = torch.ones((1,10,12))
bce = BCELoss(reduction='sum')
result2 = BCE_LOG_LOSS.forward(input_onechannel, target.float())
result3 = bce.forward(input_soft, target_onhot.float())
print(result2,result3)

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
