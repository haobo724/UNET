import csv
import glob

import numpy as np
from matplotlib import pyplot, pyplot as plt


# data = r'.\pre-epoch=71-valid_IOU=0.821612.csv'
# data_post = r'.\epoch=71-valid_IOU=0.821612.csv'
def draw():
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
def read_tensor_board(path):
    from tensorboard.backend.event_processing import event_accumulator

    # 加载日志数据
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    print(ea.scalars.Keys())
    fig=plt.figure(figsize=(6,4))
    ax1=fig.add_subplot(111)
    # train_loss=ea.scalars.Items('epoch')
    # ax1.plot([i.step for i in train_loss],[i.value for i in train_loss],label='train_loss')
    acc=ea.scalars.Items('val_loss')
    ax1.plot([i.step for i in acc],[i.value for i in acc],label='train_loss')
    # ax1.set_xlim(0)

    ax1.set_xlabel("step")
    ax1.set_ylabel("")

    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    path = r'F:\semantic_segmentation_unet\lightning_logs\version_310\events.out.tfevents.1665359849.DESKTOP-CG1LNDD.25952.0'
    read_tensor_board(path)