import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

img = plt.imread('UNET_architecture.png')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.repeat(10, 1, 1, 1)
print(img_tensor.type())
print(img_tensor.size())
print(torch.unique(img_tensor))
torchvision.utils.save_image(img_tensor, 'out.jpg')
