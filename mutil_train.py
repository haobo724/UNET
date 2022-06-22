from argparse import ArgumentParser
import numpy as np
import torchmetrics
from matplotlib import pyplot as plt
from monai.losses import DiceLoss
from model import UNET, UNET_S, UNet_PP, UNET_res, Resnet_Unet
import torch.nn as nn
import torchvision, torch
import pytorch_lightning as pl
import logging
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def mapping_color_tensor(img):
    '''
    自己写的，速度快不少，但要自己规定colormap，也可以把制定colormap拿出来单独用randint做，
    但是不能保证一个series里每次运行生成的colormap都一样，或许可以用种子点？
    反正类少还是可以考虑用这个
            '''
    # img = torch.unsqueeze(img,dim=-1)

    # img = torch.stack([img, img, img], dim=1)
    color_map = [[247, 251, 255], [171, 207, 209], [55, 135, 192]]
    for label in range(3):
        cord_1 = np.where(img[:, 0, ...] == label)
        img[cord_1[0], 0, cord_1[1], cord_1[2]] = color_map[label][0]
        img[cord_1[0], 1, cord_1[1], cord_1[2]] = color_map[label][1]
        img[cord_1[0], 2, cord_1[1], cord_1[2]] = color_map[label][2]
    if torch.is_tensor(img):
        return img
    return img.astype(int)


class mutil_train(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.__init__(hparams)
        self.loss = nn.CrossEntropyLoss()
        self.iou = torchmetrics.classification.iou.IoU(num_classes=3, absent_score=1, reduction='none').cuda()
        if hparams['model'] != 'Unet':
            self.model = UNet_PP(num_classes=3, input_channels=3).cuda()
            print('[INFO] Use Unet++')
        else:
            # self.model = UNET_S(in_channels=3, out_channels=3).cuda()
            # self.model = Resnet_Unet().cuda()
            self.model = smp.Unet(
                # encoder_depth=4,
                # decoder_channels=[512,256, 128, 64,32],
               in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,  # model output channels (number of classes in your dataset)
            ).cuda()
    def get_model_info(self):
        try:
            name=self.model.name
        except :
            name= 'Unet_S'
        return name

    def configure_optimizers(self):
        # return torch.optim.RMSprop(self.parameters(), lr=self.hparamss['lr'])
        print(self.hparams)
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.loss.forward(y_hat, y.long())

        self.log("loss", loss)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        print('============start validation==============')

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        folder = "saved_images/"
        x, y = batch
        y_hat = self(x)
        preds = torch.softmax(y_hat, dim=1)

        loss = self.loss.forward(y_hat, y.long())

        pred = preds.argmax(dim=1).float()

        self.log("val_loss", loss)

        y = y.long()
        RS_IOU = self.iou(pred.long(), y.unsqueeze(1))

        self.log("IOU0:", RS_IOU[0], prog_bar=True)
        self.log("IOU1:", RS_IOU[1], prog_bar=True)
        self.log("IOU2:", RS_IOU[2], prog_bar=True)


        torchvision.utils.save_image(torchvision.utils.make_grid(pred.unsqueeze(1),nrow=self.hparams['batch_size'],normalize=True), f"{folder}/pred_{batch_idx}.png")
        torchvision.utils.save_image(torchvision.utils.make_grid(y.unsqueeze(1).float(),nrow=self.hparams['batch_size'],normalize=True), f"{folder}/label_{batch_idx}.png")

        return {"val_loss": loss,
                "iou": (RS_IOU[1] + RS_IOU[2]) / 2, }
        # 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        # self.train_logger.info("Validatoin epoch {} ends, val_loss = {}".format(self.current_epoch, avg_loss))
        self.log('val_loss', avg_loss)
        self.log('val_Iou', avg_iou, logger=True)
        # self.log('valid_ACC', avg_acc, logger=True)
        print('============end validation==============')

    def test_step(self, batch, batch_idx, dataset_idx=None):
        folder = "saved_images/"

        def mapping_color(img):
            '''
            自己写的，速度快不少，但要自己规定colormap，也可以把制定colormap拿出来单独用randint做，
            但是不能保证一个series里每次运行生成的colormap都一样，或许可以用种子点？
            反正类少还是可以考虑用这个
                    '''
            color_map = [[247, 251, 255], [171, 207, 209], [55, 135, 192]]
            for label in range(3):
                cord_1 = np.where(img[..., 0] == label)
                img[cord_1[0], cord_1[1], 0] = color_map[label][0]
                img[cord_1[0], cord_1[1], 1] = color_map[label][1]
                img[cord_1[0], cord_1[1], 2] = color_map[label][2]
            return img.astype(int)
        x, y = batch
        y_hat = self(x)
        preds = torch.softmax(y_hat, dim=1)

        pred = preds.argmax(dim=1).float()
        img = np.stack([pred[0] for _ in range(3)], axis=-1)
        img = mapping_color(img)
        temp = np.array(torch.moveaxis(x[0], 0, 2) * 255).astype(int)
        concat = np.hstack([img, temp])
        plt.imshow(concat)
        plt.show()
