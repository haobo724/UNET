import logging
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from matplotlib import pyplot as plt

from model import UNet_PP, UNET_S, UNET
from test import AttentionUNet
from utils import FocalLoss

torch.cuda.empty_cache()


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
        self.lr = self.hparams['lr']
        self.init_val_iou = True
        # self.loss = nn.CrossEntropyLoss()
        self.loss = FocalLoss(gamma=2)
        self.iou = torchmetrics.classification.iou.IoU(num_classes=3, absent_score=1, reduction='none').cuda()
        self.model_name = hparams['model']
        if self.model_name == 'Unet-res':
            self.model = smp.Unet(
                in_channels=3,
                classes=3,
            ).cuda()
        elif self.model_name == 'Unet-vgg16':
            self.model = smp.Unet(encoder_name='vgg16_bn',
                                  in_channels=3,
                                  classes=3,  # model output channels (number of classes in your dataset)
                                  ).cuda()
        elif self.model_name == 'Unet-pp-res':
            self.model = smp.UnetPlusPlus(
                in_channels=3,
                classes=3,
            ).cuda()
        elif self.model_name == 'Unet-at-res':
            self.model = AttentionUNet(img_ch=3, output_ch=3, res=True).cuda()

        else:
            raise NameError('model name wrong')
            # self.model = UNET_S(in_channels=3, out_channels=3).cuda()
            # self.model=smp.UnetPlusPlus(in_channels=3,classes=3).cuda()
        print(f'[INFO] Use {self.model_name}')

        self.automatic_optimization = True



    def configure_optimizers(self):
        print(self.hparams)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5,
                                                                        verbose=True, threshold=0.005,
                                                                        threshold_mode='rel', cooldown=3, min_lr=1e-08,
                                                                        eps=1e-08),
                "monitor": "val_Iou",
                "frequency": self.trainer.check_val_every_n_epoch,
                "interval": "epoch",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        if self.init_val_iou:
            print("\n initing val iou to 0 for metric tracking")
            self.log("val_Iou", 0)
            self.init_val_iou = False

    def get_model_info(self):
        return self.model_name

    def training_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.loss.forward(y_hat, y.long())

        self.log("train_loss", loss)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        print('============start validation==============')

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch
        y_hat = self(x)

        loss = self.loss.forward(y_hat, y.long())
        preds = torch.softmax(y_hat, dim=1)

        pred = preds.argmax(dim=1).float()

        self.log("val_loss", loss)

        y = y.long()
        RS_IOU = self.iou(pred.long(), y.unsqueeze(1))

        self.log("IOU0:", RS_IOU[0], prog_bar=True)
        self.log("IOU1:", RS_IOU[1], prog_bar=True)
        self.log("IOU2:", RS_IOU[2], prog_bar=True)
        # folder = "saved_images/"
        # torchvision.utils.save_image(
        #     torchvision.utils.make_grid(pred.unsqueeze(1), nrow=self.hparams['batch_size'], normalize=True),
        #     f"{folder}/pred_{batch_idx}.png")
        # torchvision.utils.save_image(
        #     torchvision.utils.make_grid(y.unsqueeze(1).float(), nrow=self.hparams['batch_size'], normalize=True),
        #     f"{folder}/label_{batch_idx}.png")

        return {"val_loss": loss,
                "iou": (RS_IOU[1] + RS_IOU[2]) / 2, }
        # 'acc': acc}

    def validation_epoch_end(self, outputs):
        # sch = self.lr_schedulers()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        # self.train_logger.info("Validatoin epoch {} ends, val_loss = {}".format(self.current_epoch, avg_loss))
        self.log('val_loss', avg_loss)
        self.log('val_Iou', avg_iou, prog_bar=True,logger=True)
        # self.log('valid_ACC', avg_acc, logger=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True,logger=True)

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        # if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     sch.step(self.trainer.callback_metrics["val_Iou"])
        #     print('val_Iou',self.trainer.callback_metrics["val_Iou"])
        #
        #     # print('real lr',self.optimizers().param_groups[0]['lr'])
        #     # print('all',self.optimizers().param_groups[0])
        #     # print('all lr_schedulers',lr)
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


class unet_train(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        try:
            if hparams['mode_size'] == 32:
                print('small size')
                self.model = UNET_S(in_channels=3, out_channels=1).cuda()
            elif hparams['mode_size'] == 16:
                print('Xsmall size')
                self.model = UNET_S(in_channels=3, out_channels=1, features=[16, 32, 64, 128]).cuda()
            else:
                self.model = UNET(in_channels=3, out_channels=1).cuda()
        except:
            self.model = UNET_S(in_channels=3, out_channels=1).cuda()
        if hparams['model'] != 'Unet':
            self.model = UNet_PP(num_classes=1, input_channels=3).cuda()
            print('Use Unet++')
        self.weights = torch.tensor(np.array([0.2, 0.8])).float()
        self.loss = nn.BCEWithLogitsLoss()
        self.train_logger = logging.getLogger(__name__)
        self.learning_rate = None
        self.hparamss = hparams
        self.save_hyperparameters()
        self.test_save = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch

        y_hat = self(x)

        loss = self.loss.forward(y_hat, y.unsqueeze(dim=1))
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        print('============start validation==============')
        folder = "saved_images/"
        x, y = batch
        y_hat = self(x)
        loss = self.loss.forward(y_hat, y.unsqueeze(dim=1))

        self.log("loss", loss, on_epoch=True)

        preds = torch.sigmoid(y_hat.squeeze())
        pred = (preds > 0.6).float()

        iou = torchmetrics.IoU(num_classes=2, absent_score=1, reduction='none').cuda()
        # validation_recall = torchmetrics.Recall(average='macro', mdmc_average='samplewise', multiclass=True,
        #                                         num_classes=2).cuda()
        # validation_precision = torchmetrics.Precision(average='macro', mdmc_average='samplewise', multiclass=True,
        #                                               num_classes=2).cuda()
        validation_ACC = torchmetrics.Accuracy().cuda()
        # pred=preds.int().unsqueeze(dim=1)
        y = y.long()
        RS_IOU = iou(pred.unsqueeze(1), y.unsqueeze(1))
        # RS_recall = validation_recall(pred.unsqueeze(1), y.unsqueeze(1))
        # RS_precision = validation_precision(pred.unsqueeze(1), y.unsqueeze(1))
        acc = validation_ACC(pred.unsqueeze(1), y.unsqueeze(1))

        self.log("IOU:", RS_IOU[1], prog_bar=True)

        # torchvision.utils.save_image(
        #     pred.unsqueeze(dim=1).cpu(), f"{folder}/pred_{batch_idx}.png"
        # )
        #
        # torchvision.utils.save_image(y.float().unsqueeze(dim=1).cpu(), f"{folder}/label_{batch_idx}.png")

        return {"loss": loss,
                "iou": RS_IOU[1],
                'acc': acc}

    def test_step(self, batch, batch_idx, dataset_idx=None):
        folder = "saved_images/"
        x, y = batch
        y_hat = self(x)
        preds = torch.sigmoid(y_hat.squeeze())
        preds = (preds > 0.8).float()

        iou = pl.metrics.IoU(num_classes=2, absent_score=1, reduction='none')
        validation_recall = pl.metrics.Recall(average='macro', mdmc_average='samplewise', num_classes=2)
        validation_precision = pl.metrics.Precision(average='macro', mdmc_average='samplewise', num_classes=2)
        pred = preds.int().unsqueeze(dim=1).cpu()
        target = y.clone().int().unsqueeze(dim=1).cpu()
        RS_IOU = iou(pred, target)
        RS_recall = validation_recall(pred, target)
        RS_precision = validation_precision(pred, target)

        self.log("IOU:", RS_IOU, prog_bar=True)
        self.log("RS_recall:", RS_recall, prog_bar=True)
        self.log("RS_precision:", RS_precision, prog_bar=True)

        if self.test_save:
            torchvision.utils.save_image(
                preds.unsqueeze(1), f"{folder}/pred_{batch_idx}.png"
            )

            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/label_{batch_idx}.png")

        return {
            "iou": RS_IOU}

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train/loss', avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        # self.train_logger.info("Validatoin epoch {} ends, val_loss = {}".format(self.current_epoch, avg_loss))
        self.log('val_loss', avg_loss)
        self.log('val_Iou', avg_iou, logger=True)
        # self.log('valid_IOU', avg_iou, logger=True)
        # self.log('valid_ACC', avg_acc, logger=True)
        print('============end validation==============')

    #
    # def test_epoch_end(self, outputs):
    #
    #     avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
    #     self.train_logger.info("Test epoch {} ends, val_loss = {}".format(self.current_epoch, avg_iou))
    #     self.log('valid/IOU', avg_iou, logger=True)

    def configure_optimizers(self):
        # return torch.optim.RMSprop(self.parameters(), lr=self.hparamss['lr'])
        return torch.optim.Adam(self.parameters(), lr=self.hparamss['lr'])
        # return torch.optim.Adam(self.parameters(),lr=self.learning_rate)
