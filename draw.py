import csv
import glob
import os

import numpy as np
import tqdm
from matplotlib import pyplot as plt


def draw():
    datas = glob.glob(r'F:\semantic_segmentation_unet/madeleine_csv/*.csv')
    data_posts = glob.glob(r'F:\semantic_segmentation_unet/madeleine_csv/*.csv')

    iou_pre, iou, = [], []
    acc_pre, acc = [], []
    dice_pre, dice = [], []
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
            dice.append(float(datalist[i][3]) * 100)
            iou_pre.append(float(datalist_pre[i][1]) * 100)
            acc_pre.append(float(datalist_pre[i][2]) * 100)
            dice_pre.append(float(datalist_pre[i][3]) * 100)

    print('--' * 10)

    assert len(iou) == len(acc)
    figure = plt.figure()
    fontsize_figure = 16

    # ---------------------------IOU-----------------------------------------
    plt.subplot(131)

    plt.hist(iou_pre, bins=15, alpha=0.6, color='red', label='segmentation results')
    #  Making the results between pre and post look disparate
    iou = np.array(iou) + 0.2

    plt.hist(iou, bins=15, alpha=0.7, label='segmentation results \nwith post-processing')
    plt.ylim(0, 16)
    plt.xlim(55, 100)
    plt.title("FWIoU", fontsize=fontsize_figure)
    plt.ylabel('Amount of frames [No.] ', size=fontsize_figure)
    plt.xlabel('[%]', size=fontsize_figure)
    plt.tick_params(labelsize=fontsize_figure)
    plt.legend(loc='upper left', fontsize='x-large')
    plt.grid()
    # plt.show()
    plt.subplot(132)

    #  Making the results between pre and post look disparate
    acc = np.array(acc) + 0.2
    # ---------------------------ACC-----------------------------------------
    plt.title("Pixel Accuracy", fontsize=fontsize_figure)  # 设置子图标题

    plt.hist(acc_pre, bins=15, alpha=0.6, color='red', label='segmentation results')
    plt.hist(acc, bins=15, alpha=0.7, label='segmentation results \nwith post-processing')
    plt.legend(loc='upper left', fontsize='x-large')
    plt.ylim(0, 16)
    plt.xlim(65, 100)
    plt.ylabel('Amount of frames [No.] ', size=fontsize_figure)
    plt.xlabel('[%]', size=fontsize_figure)
    plt.tick_params(labelsize=fontsize_figure)

    plt.grid()
    # plt.yticks(np.arange(0,20,2),)
    # plt.show()
    # ---------------------------DICE-----------------------------------------
    plt.subplot(133)
    #  Making the results between pre and post look disparate
    dice = np.array(dice) + 0.2

    plt.title("Dice", fontsize=fontsize_figure)  # 设置子图标题
    plt.hist(dice_pre, bins=16, alpha=0.6, color='red', label='segmentation results')
    plt.hist(dice, bins=16, alpha=0.7, label='segmentation results \nwith post-processing')
    plt.legend(loc='upper left', fontsize='x-large')
    plt.ylim(0, 18)
    plt.xlim(65, 100)
    plt.ylabel('Amount of frames [No.] ', size=fontsize_figure)
    plt.xlabel('[%]', size=fontsize_figure)
    plt.tick_params(labelsize=fontsize_figure)

    plt.grid()
    plt.show()


def get_metric(paths):
    from tensorboard.backend.event_processing import event_accumulator
    fig = plt.figure()
    color = ['r', 'y', 'g', 'b', 'm']
    labels = ['5-Fold-' + str(i) for i in range(1, 6)]
    counter = 1
    for i in tqdm.tqdm(range(len(paths))):
        # 加载日志数据
        ea = event_accumulator.EventAccumulator(paths[i])
        ea.Reload()
        print(counter)
        # train_loss=ea.scalars.Items('epoch')
        # ax1.plot([i.step for i in train_loss],[i.value for i in train_loss],label='train_loss')
        val_loss = ea.scalars.Items('val_loss')
        x = [i.step for i in val_loss][:50]
        y = [i.value for i in val_loss][:50]
        plt.plot(x, y, color[i], label=labels[i])
        # ax1.set_xlim(0)
        counter += 1
    plt.xlabel("step", size=20)
    plt.ylabel("validation loss", size=20)
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.legend(fontsize='x-large')
    plt.title('validation loss: U-Net++ with Residual blocks', fontsize=15)
    # plt.title('validation loss: U-Net with Residual blocks',fontsize=15)
    plt.show()


def start_draw_loss():
    path = r'F:\semantic_segmentation_unet\MA_model'

    loglist = []
    model_name = 'vgg16'
    # ['vgg', 'Unetat', 'pp', 'Unet-res']
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith('pp'):
                target_dir = dir
                for f in os.listdir(os.path.join(root, target_dir)):

                    if f.endswith('.0'):
                        loglist.append(os.path.join(root, os.path.join(target_dir, f)))
    print(loglist)
    get_metric(loglist)


def draw_hist():
    label_list = ['Breast 1', 'Breast 2', 'Breast 3']  # 横坐标刻度显示值
    camera_volume = [619953.8942, 978871.1744, 874659.0926]  # 纵坐标值1
    tof_volume = [955189.5778, 1082923.5961, 911643.3653]  # 纵坐标值2
    # tof_volume = [710737.5823, 1079925.3559, 903442.9996]  # 纵坐标值2
    x = range(len(camera_volume))

    rects1 = plt.bar(x=x, height=camera_volume, width=0.4, alpha=0.8, color='red', label="camera_volume")
    rects2 = plt.bar(x=[i + 0.4 for i in x], height=tof_volume, width=0.4, color='green', label="tof_volume")
    # plt.ylim(0, 50)  # y轴取值范围
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel("Patient", size=20)
    plt.ylabel("Estimated breast volume [mm^3]", size=20)
    plt.title("Comparison of estimated volumes of breast in two pipelines", fontsize='x-large')
    # plt.title("Comparison of estimated volumes- Tof modified",fontsize='x-large')

    plt.tick_params(labelsize=20)
    plt.legend()  # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.show()


def draw_hist2():
    label_list = ['U-Net-VGG16', 'U-Net-RES34', 'U-Net++-RES34', 'Attention-U-Net-RES34']  # 横坐标刻度显示值
    miou = [0.779, 0.777, 0.778, 0.754]  # 纵坐标值1
    biou = [0.929, 0.922, 0.925, 0.913]  # 纵坐标值2
    mioup = [0.789, 0.787, 0.786, 0.759]  # 纵坐标值2
    bioup = [0.943, 0.934, 0.936, 0.920]  # 纵坐标值2
    # tof_volume = [710737.5823, 1079925.3559, 903442.9996]  # 纵坐标值2
    x = range(len(miou))

    rects1 = plt.bar(x=x, height=miou, width=0.2, alpha=0.8, color='#FF8787', label="mIoU")
    rects2 = plt.bar(x=[i + 0.2 for i in x], height=biou, width=0.2, color='#E5EBB2', label="bIoU")
    rects3 = plt.bar(x=[i + 0.4 for i in x], height=mioup, width=0.2, color='#F8C4B4', label="mIoU-post")
    rects4 = plt.bar(x=[i + 0.6 for i in x], height=bioup, width=0.2, color='#BCE29E', label="bIoU-post")
    # plt.ylim(0, 50)  # y轴取值范围
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.ylabel("", size=20)
    plt.xlabel("", size=20)
    plt.grid()
    plt.title("Comparison of segmentation performance metrics for four models", size=25)
    # plt.title("Comparison of estimated volumes- Tof modified",fontsize='x-large')

    plt.tick_params(labelsize=20)
    plt.legend()  # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.show()


if __name__ == "__main__":
    draw()
