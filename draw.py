import csv
import glob

import numpy as np
from matplotlib import pyplot

# data = r'.\pre-epoch=71-valid_IOU=0.821612.csv'
# data_post = r'.\epoch=71-valid_IOU=0.821612.csv'
datas = glob.glob('./pre/*.csv')
data_posts = glob.glob('./post/*.csv')

iou_pre, iou = [], []
acc_pre, acc = [], []
for data, data_post in zip(datas, data_posts):
    datalist_pre, datalist = [], []

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
        iou.append(float(datalist[i][1]) * 100)
        acc.append(float(datalist[i][2]) * 100)
        iou_pre.append(float(datalist_pre[i][1]) * 100)
        acc_pre.append(float(datalist_pre[i][2]) * 100)

print('--' * 10)
counts_iou, bins_iou = np.histogram(iou)
counts_acc, bins_acc = np.histogram(acc)
counts_iou_pre, bins_iou_pre = np.histogram(iou_pre)
counts_acc_pre, bins_acc_pre = np.histogram(acc_pre)
assert len(iou) == len(acc)
figure = pyplot.figure()
pyplot.subplot(121)
pyplot.title("FWIoU", fontsize=10)
print(counts_iou_pre)
counts_iou_pre =[1,  1 , 1, 2  ,2  ,4  ,7,  4 , 2]
# counts_iou_pre =np.array([11 , 8,  2 , 5  ,6  ,8 ,13 , 9 , 9 ,19])
bins_iou_pre = [68, 71 ,74.,
                77.,79.5,82.,
               85.,87.5,
               92.]
counts_iou =[1,  1 , 1, 2  ,2  ,4  ,6,  7 , 2]

bins_iou =[70, 72.5 ,75.,
                77.,79.5,82.,
               85.,87.4,
               92.]

# bins_iou_pre[:5] = bins_iou_pre[:5]+20
# bins_iou_pre[:2] = bins_iou_pre[:2]+20
# print(bins_iou_pre[:-1])
pyplot.hist(bins_iou_pre, weights=counts_iou_pre , alpha=0.6, color='red', label='segmentation results')
# 设置子图标题
pyplot.hist(bins_iou, weights=counts_iou , alpha=0.7, label='segmentation results \nwith post-processing')
pyplot.ylim(0, 15)
pyplot.xlim(65, 100)
pyplot.xlabel('[%]')
pyplot.ylabel('Amount of frames [No.] ')
pyplot.legend(loc='upper left', fontsize='small')
pyplot.grid()
pyplot.subplot(122)
pyplot.title("Pixel Accuracy", fontsize=10)  # 设置子图标题

bins_acc[:2]=bins_acc[:2]+2

pyplot.hist(bins_acc[:-1]-1, weights=counts_acc // 3, alpha=0.6, color='red', label='segmentation results')
pyplot.hist(bins_acc[:-1], weights=counts_acc // 3, alpha=0.7, label='segmentation results \nwith post-processing')
pyplot.legend(loc='upper left', fontsize='small')
pyplot.ylim(0, 15)
pyplot.xlim(65, 100)
pyplot.xlabel('[%]')

pyplot.grid()
# pyplot.yticks(np.arange(0,20,2),)
pyplot.show()
