import glob
import os
import shutil

import cv2
from tqdm import tqdm


def split(path, output_path, sample_rate=10,only_one_frame = False,idx=[]):
    save_name = os.path.basename(path)
    output_dir = os.path.join(output_path, save_name)
    print(f'[INFO] Video is {save_name}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stream = cv2.VideoCapture(path)
    frame_nr = 0
    with tqdm(total=stream.get(cv2.CAP_PROP_FRAME_COUNT)) as bar:  # total表示预期的迭代次数
        if only_one_frame:
            while True:
                ret, frame = stream.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_nr == stream.get(cv2.CAP_PROP_FRAME_COUNT)//2:
                    cv2.imwrite(os.path.join(output_dir, '{}_{:0>3}.jpg'.format(save_name, frame_nr)), frame)
                bar.update(stream.get(cv2.CAP_PROP_FRAME_COUNT)//2)
                frame_nr += 1

            print('ONLY ONE FRAME DONE')
        else:
            if not len(idx):
                while True:
                    ret, frame = stream.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

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

                    if frame_nr in idx :
                        cv2.imwrite(os.path.join(output_dir, '{}_{:0>3}.jpg'.format(save_name, frame_nr)), frame)
                    bar.update(1)
                    frame_nr += 1

            print('DONE')


def rename(file,prefix=''):
    print(f'origin name = {file}')
    path_name = os.path.dirname(file)
    file_name = os.path.basename(file)
    if file_name.split('_')[0][0] !='p':
        file_name = file_name.split('_')[1:]
        connect =''
        file_name=connect.join(file_name)
    new_file_name = os.path.join(path_name,prefix+file_name)
    print(new_file_name)
    os.rename(file,new_file_name)
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
                cv2.imshow('file_path',frame)
                k = cv2.waitKey()
                if k == ord('y') or k ==ord('Y'):
                    shutil.copy(file_path, moved_path)
                    cv2.destroyAllWindows()
                else:
                    cv2.destroyAllWindows()
                    continue
            if os.path.isdir(file_path):  # 如果目录，就递归子目录
                copy_file(file_path, moved_path)



if __name__ == '__main__':
    path = r'F:\semantic_segmentation_unet\collected_data\2G'
    output_path = 'output_test\picked'
    print(f'[INFO] input path is {path}')
    print(f'[INFO] output path is {os.path.abspath(output_path)}')
    videos = glob.glob(os.path.join(path, '*_top.mp4'))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in videos:
        print(i)
        # rename(i,'6G_')
        split(i, output_path,sample_rate=1,only_one_frame=True,idx=[0,94,101,128,227])
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    copy_file(output_path,dataset_path)
