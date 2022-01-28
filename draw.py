from matplotlib import pyplot
import csv

data = r'.\pre-epoch=71-valid_IOU=0.821612.csv'
data_post = r'.\epoch=71-valid_IOU=0.821612.csv'

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

iou_pre ,iou = [],[]
acc_pre,acc = [],[]
for i in range(len(datalist) - 1):
    iou.append(float(datalist[i][1])*100)
    acc.append(float(datalist[i][2])*100)
    iou_pre.append(float(datalist_pre[i][1])*100)
    acc_pre.append(float(datalist_pre[i][2])*100)

print('--' * 10)

assert len(iou) == len(acc)
figure=pyplot.figure()
pyplot.subplot(121)
pyplot.title("FWIoU", fontsize=10)
pyplot.hist(iou_pre,alpha = 0.6,color='red',label='segmentation results')
#设置子图标题
pyplot.hist(iou,histtype='bar',alpha = 0.8,label='segmentation results \nwith post-processing')
pyplot.ylim(0,14)
pyplot.xlim(55, 100)
pyplot.xlabel('[%]')
pyplot.ylabel('Amount of frames [No.] ')
pyplot.legend(loc='upper left',fontsize='small' )
pyplot.grid()
pyplot.subplot(122)
pyplot.title("Pixel Accuracy", fontsize=10)             #设置子图标题
pyplot.hist(acc_pre,alpha = 0.6,color='red',label='segmentation results')
pyplot.hist(acc,alpha = 0.8,label='segmentation results \nwith post-processing')
pyplot.legend(loc='upper left',fontsize='small' )
pyplot.ylim(0,14)
pyplot.xlim(70, 100)
pyplot.xlabel('[%]')

pyplot.grid()
pyplot.yticks([0,2,4,6,8,10,12,14], [])

pyplot.show()
