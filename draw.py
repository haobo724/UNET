from matplotlib import pyplot
import csv

data = r'.\epoch=71-valid_IOU=0.821612.csv'
datalist = []
with open(data) as f:
    inhalt = csv.reader(f)

    for idx, i in enumerate(inhalt):
        if idx == 0:
            continue
        datalist.append(i)
        print(i)

iou = []
acc = []
for i in range(len(datalist) - 1):
    iou.append(float(datalist[i][1])*100)
    acc.append(float(datalist[i][2])*100)
print('--' * 10)
print(iou)
print(acc)
assert len(iou) == len(acc)
pyplot.figure()
pyplot.subplot(121)
pyplot.title("FWIoU [%]", fontsize=10)             #设置子图标题
pyplot.hist(iou,rwidth=0.7)
pyplot.subplot(122)
pyplot.title("Pixel Accuracy [%]", fontsize=10)             #设置子图标题
pyplot.hist(acc,rwidth=0.7)
pyplot.show()
