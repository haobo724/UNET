import csv
import glob
import os

import numpy as np
from matplotlib import pyplot, pyplot as plt


# data = r'.\pre-epoch=71-valid_IOU=0.821612.csv'
# data_post = r'.\epoch=71-valid_IOU=0.821612.csv'
def draw():
    datas = glob.glob('F:\semantic_segmentation_unet\MA_CSV\selected/*.csv')
    data_posts = glob.glob('F:\semantic_segmentation_unet\MA_CSV\selected/*.csv')

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
    # pyplot.subplot(121)
    pyplot.title("bIoU", fontsize=30)


    # pyplot.hist(bins_iou_pre[1:], weights=counts_iou_pre , alpha=0.6, color='red', label='segmentation results')
    pyplot.hist(iou_pre ,bins=15, alpha=0.6, color='red', label='segmentation results')
    # 设置子图标题
    iou=np.array(iou)+0.2

    pyplot.hist(iou,bins=15,  alpha=0.7, label='segmentation results \nwith post-processing')
    pyplot.ylim(0, 12)
    pyplot.xlim(55, 100)
    pyplot.xlabel('[%]')
    pyplot.ylabel('Amount of frames [No.] ')
    pyplot.legend(loc='upper left', fontsize='x-large')
    pyplot.grid()
    pyplot.show()
    # pyplot.subplot(122)
    pyplot.title("Pixel Accuracy", fontsize=30)  # 设置子图标题

    acc=np.array(acc)-0.2

    pyplot.hist(acc_pre, bins=15,alpha=0.6, color='red', label='segmentation results')
    pyplot.hist(acc, bins=15,alpha=0.7, label='segmentation results \nwith post-processing')
    pyplot.legend(loc='upper left', fontsize='x-large')
    pyplot.ylim(0, 16)
    pyplot.xlim(65, 100)
    pyplot.xlabel('[%]')

    pyplot.grid()
    # pyplot.yticks(np.arange(0,20,2),)
    pyplot.show()
def get_metric(paths):
    from tensorboard.backend.event_processing import event_accumulator
    fig = plt.figure()
    color = ['r','y','g','b','m']
    labels = ['5-Fold-'+str(i) for i in range(1,6)]
    for i in range(len(paths)):

    # 加载日志数据
        ea = event_accumulator.EventAccumulator(paths[i])
        ea.Reload()
        # train_loss=ea.scalars.Items('epoch')
        # ax1.plot([i.step for i in train_loss],[i.value for i in train_loss],label='train_loss')
        val_loss=ea.scalars.Items('val_loss')
        x= [i.step for i in val_loss][:50]
        y = [i.value for i in val_loss][:50]
        plt.plot(x,y,color[i],label=labels[i])
        # ax1.set_xlim(0)

    plt.xlabel("step")
    plt.ylabel("validation loss")
    plt.grid()
    plt.legend()
    plt.title('validation loss: U-Net with Residual blocks')
    plt.show()

def start_draw_loss():
    path = r'F:\semantic_segmentation_unet\lightning_logs'

    loglist = []
    model_name= 'vgg16'
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith('Unetres'):
                target_dir =dir
                for f in os.listdir(os.path.join(root, target_dir)):

                    if f.endswith('.0'):

                        loglist.append(os.path.join(root,os.path.join(target_dir, f)))
    print(loglist)
    get_metric(loglist)
if __name__ == "__main__":
    draw()