import glob
import os
import shutil

import cv2
from tqdm import tqdm


def split(path, output_path, only_one_frame=False, idx=[], sample_rate=10):
    '''
    Priority:
    1.only_one_frame (set True is active)
    2.idx (list is not empty then is active)
    3.sample_rate (always active ,default mode)

    '''
    save_name = os.path.basename(path)
    output_dir = os.path.join(output_path, save_name)
    print(f'[INFO] Video is {save_name}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stream = cv2.VideoCapture(path)
    frame_nr = 0
    # now is choosed at 2/3 of frame number ,i.e 75. frame when there is 100 frames
    frame_nr_idx =stream.get(cv2.CAP_PROP_FRAME_COUNT) * 2 // 3
    with tqdm(total=stream.get(cv2.CAP_PROP_FRAME_COUNT)) as bar:
        if only_one_frame:
            while True:
                ret, frame = stream.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_nr == frame_nr_idx:
                    cv2.imwrite(os.path.join(output_dir, '{}_{:0>3}.jpg'.format(save_name, frame_nr)), frame)
                bar.update(stream.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
                frame_nr += 1

            print('ONLY ONE FRAME DONE')
        else:
            if not len(idx):
                while True:
                    ret, frame = stream.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if frame_nr % sample_rate == 0:
                        cv2.imwrite(os.path.join(output_dir, '{}_{:0>3}.jpg'.format(save_name, frame_nr)), frame)
                    bar.update(1)
                    frame_nr += 1
            else:
                while True:
                    ret, frame = stream.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if frame_nr in idx:
                        cv2.imwrite(os.path.join(output_dir, '{}_{:0>3}.jpg'.format(save_name, frame_nr)), frame)
                    bar.update(1)
                    frame_nr += 1

            print('DONE')


def rename(file, prefix=''):
    '''

    rename the video in order to distinguish between the videos of different acquisition groups
    because everytime the recorded video is saved as patient_0_top.mp4 ....patient_n_top.mp4
    '''
    print(f'origin name = {file}')
    path_name = os.path.dirname(file)
    file_name = os.path.basename(file)
    if file_name.split('_')[0][0] != 'p':
        file_name = file_name.split('_')[1:]
        connect = ''
        file_name = connect.join(file_name)
    new_file_name = os.path.join(path_name, prefix + file_name)
    print(new_file_name)
    os.rename(file, new_file_name)
    print(new_file_name)


def copy_file(orgin_path, moved_path):
    '''

    :param orgin_path:
    :param moved_path:
    :return: copy files from origin path to target path
    '''
    print('press y or Y to comfirm ,otherwise skip')
    if os.path.isfile(orgin_path):
        shutil.move(orgin_path, moved_path)
    else:
        dir_files = os.listdir(orgin_path)  # 得到该文件夹下所有的文件
        for file in dir_files:
            file_path = os.path.join(orgin_path, file)  # 路径拼接成绝对路径
            if os.path.isfile(file_path):  # 如果是文件，就打印这个文件路径
                frame = cv2.imread(file_path)
                cv2.imshow('file_path', frame)
                k = cv2.waitKey()
                if k == ord('y') or k == ord('Y'):
                    shutil.copy(file_path, moved_path)
                    cv2.destroyAllWindows()
                else:
                    cv2.destroyAllWindows()
                    continue
            if os.path.isdir(file_path):  # 如果目录，就递归子目录
                copy_file(file_path, moved_path)


if __name__ == '__main__':
    video_path = r'F:\semantic_segmentation_unet\collected_data\7G'
    splited_frame_saved_path = 'output_test\picked'
    if not os.path.exists(splited_frame_saved_path):
        os.makedirs(splited_frame_saved_path)
    print(f'[INFO] input path is {video_path}')
    print(f'[INFO] splited_frame_saved_path is {os.path.abspath(splited_frame_saved_path)}')

    # to choose only then top video , can changed as *_bot.mp4 to choose the bot video or *.mp4 to choose all video
    videos = glob.glob(os.path.join(video_path, '*_top.mp4'))

    for video in videos:
        # rename(i,'7G_')
        split(video, splited_frame_saved_path, only_one_frame=False, idx=[], sample_rate=10)

    # choose the file 2. part
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    copy_file(splited_frame_saved_path, dataset_path)
