import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_dir = r'F:\semantic_segmentation_unet\data\test_new'
# TEST_MASK_DIR = r'F:\opencv\socket_demo\export'
mask_dir = r'F:\semantic_segmentation_unet\data\test_new_mask'
def rename(file, prefix=''):
    '''

    rename the video in order to distinguish between the videos of different acquisition groups
    because everytime the recorded video is saved as patient_0_top.mp4 ....patient_n_top.mp4
    '''
    print(f'origin name = {file}')
    path_name = os.path.dirname(file)
    file_name = os.path.basename(file)
    prefix='-labels'
    if prefix in file_name:
        return
    basename = file_name.split('_m')[0]
    print(basename)
    newname = basename+prefix
    new_file_name = os.path.join(path_name, newname+'.tiff')
    os.rename(file, new_file_name)
    print(new_file_name)
def convert_BGR(file):
    print(f'origin name = {file}')
    img = cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(file,img)
def rot_img(file,key):
    file_name = os.path.basename(file)
    if file_name[0] in key:
        return
    else:
        print(f'{file} rotated' )
        img = cv2.imread(file)

        # plt.imshow(img)
        # plt.show()
        img = np.rot90(img,2)
        cv2.imwrite(file, img)

if __name__ == "__main__":

    filename_mask = sorted(glob.glob(os.path.join(mask_dir, "*.tiff")))
    filename_img = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    for i ,j in zip(filename_img,filename_mask):
        rot_img(i,key=['p','W'])
        rot_img(j,key=['p','W'])
