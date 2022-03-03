import csv,glob

import numpy as np
from matplotlib import pyplot

# data = r'.\pre-epoch=71-valid_IOU=0.821612.csv'
# data_post = r'.\epoch=71-valid_IOU=0.821612.csv'
datas = glob.glob('./pre/*.csv')
data_posts = glob.glob('./post/*.csv')

iou_pre ,iou = [],[]
acc_pre,acc = [],[]
for data,data_post in zip(datas,data_posts):
    datalist_pre,datalist = [],[]

    with open(data_post) as f:
        inhalt = csv.reader(f)

        for idx, i in enumerate(inhalt):
            if idx == 0:
                continue
            datalist.append(i)

    with open(data) as f:
        inhalt = csv.reader(f)

        for idx, i in enumerate(inhalt):
            if idx == 0:
                continue
            datalist_pre.append(i)

    for i in range(len(datalist) - 1):
        iou.append(float(datalist[i][1])*100)
        acc.append(float(datalist[i][2])*100)
        iou_pre.append(float(datalist_pre[i][1])*100)
        acc_pre.append(float(datalist_pre[i][2])*100)

print('--' * 10)
print(np.min(acc_pre),np.min(acc))
counts_iou, bins_iou = np.histogram(iou)
counts_acc, bins_acc = np.histogram(acc)
counts_iou_pre, bins_iou_pre = np.histogram(iou_pre)
counts_acc_pre, bins_acc_pre = np.histogram(acc_pre)
assert len(iou) == len(acc)
figure=pyplot.figure()
pyplot.subplot(121)
pyplot.title("FWIoU", fontsize=10)
pyplot.hist(bins_iou_pre[:-1],weights=counts_iou_pre//5,alpha = 0.6,color='red',label='segmentation results')
#设置子图标题
pyplot.hist(bins_iou[:-1],weights=counts_iou//5,alpha = 0.7,label='segmentation results \nwith post-processing')
pyplot.ylim(0,15)
pyplot.xlim(65, 100)
pyplot.xlabel('[%]')
pyplot.ylabel('Amount of frames [No.] ')
pyplot.legend(loc='upper left',fontsize='small' )
pyplot.grid()
pyplot.subplot(122)
pyplot.title("Pixel Accuracy", fontsize=10)             #设置子图标题
pyplot.hist(bins_acc_pre[:-1],weights=counts_acc_pre//5,alpha = 0.6,color='red',label='segmentation results')
pyplot.hist(bins_acc[:-1],weights=counts_acc//5,alpha = 0.7,label='segmentation results \nwith post-processing')
pyplot.legend(loc='upper left',fontsize='small' )
pyplot.ylim(0,15)
pyplot.xlim(65, 100)
pyplot.xlabel('[%]')

pyplot.grid()
# pyplot.yticks(np.arange(0,20,2),)
print(np.arange(0,20,2))
pyplot.show()
