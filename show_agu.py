import glob
import os.path

import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt

IMAGE_HEIGHT = 480  # 1096 originally  0.25
IMAGE_WIDTH = 640  # 1936 originally

def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = 4

    for i in range(1, len(images)+1):
        if bboxes is not None:
            img = visualize_bbox(images[i - 1], bboxes[i - 1], class_name="Elon")
        else:
            img = images[i-1]
        ax = fig.add_subplot(rows, columns, i)
        if i <=len(images)//4:
            ax.title.set_text('Original input')
        elif i >len(images)//4 and i <=len(images)*2//4:
            ax.title.set_text('Original mask')

        elif i >len(images)*2//4 and i <=len(images)*3//4:
            ax.title.set_text('Augmented input')

        else:
            ax.title.set_text('Augmented mask')

        plt.imshow(img)
    plt.show()
def mapping_color(img):
    '''
    自己写的，速度快不少，但要自己规定colormap，也可以把制定colormap拿出来单独用randint做，
    但是不能保证一个series里每次运行生成的colormap都一样，或许可以用种子点？
    反正类少还是可以考虑用这个
            '''
    color_map = [[247, 251, 255], [1, 207, 209], [2, 255, 0]]
    for label in range(3):
        cord_1 = np.where(img[..., 0] == label)
        img[cord_1[0], cord_1[1], 0] = color_map[label][0]
        img[cord_1[0], cord_1[1], 1] = color_map[label][1]
        img[cord_1[0], cord_1[1], 2] = color_map[label][2]
    return img.astype(int)

# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=5):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img



train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        A.ColorJitter(brightness=0.5,contrast=0.5,hue=0,saturation=0, p=1),
        A.Rotate(limit=10, p=0.4),
        A.HorizontalFlip(p=0.2),  # A.VerticalFlip(p=0.2),

    ],
)
input_path = glob.glob(r'C:\Users\94836\Desktop\MasterTheis\mask\*.png')
mask_path = glob.glob(r'C:\Users\94836\Desktop\MasterTheis\mask\*.tiff')
img_list = []
agu_list = []
mask_list = []
mask_list2 = []
# idx_list = [60,45,100]
idx_list = [0,1,2]
for i in idx_list:
    input_image = cv2.imread(input_path[i])
    # basename = os.path.basename(input_path[i])
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    # input_image = np.rot90(input_image,2)
    input_image = cv2.resize(input_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    cv2.imwrite(input_path[i], input_image)

    # agu = train_transform(image=input_image,mask=input_mask)
    # mask,trans_image =agu['mask'],agu['image']
    # mask = mapping_color(mask)
    # con = np.concatenate((input_image,trans_image),axis=0)
    img_list.append(input_image)
    # agu_list.append(trans_image)
    # mask_list2.append(mask)
    # cv2.imshow('hi',con)
    # cv2.waitKey()
input_mask = cv2.imread(mask_path[0])
input_mask = cv2.resize(input_mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
input_mask = np.rot90(input_mask,2)
input_mask = mapping_color(input_mask)
mask_list.append(input_mask)

add_wegit =cv2.addWeighted(input_mask.astype(np.uint8),0.5,img_list[0].astype(np.uint8),0.5,1)
add_wegit = cv2.cvtColor(add_wegit,cv2.COLOR_BGR2RGB)
# add_wegit = np.rot90(add_wegit,2)

cv2.imwrite(r'C:\Users\94836\Desktop\MasterTheis\mask\out.jpg',add_wegit)
plot_examples(img_list+mask_list+agu_list+[add_wegit])
