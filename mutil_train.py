from argparse import ArgumentParser

import numpy as np
import torchmetrics
from model import UNET, UNET_S
import torch.nn as nn
import torchvision,torch
import pytorch_lightning as pl
import logging

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
        # self.model =  UNet_PP(num_classes=1, input_channels=3).cuda()
        self.weights = torch.tensor(np.array([0.5, 0.5])).float()
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


        torchvision.utils.save_image(
            pred.unsqueeze(dim=1).cpu(), f"{folder}/pred_{batch_idx}.png"
        )

        torchvision.utils.save_image(y.float().unsqueeze(dim=1).cpu(), f"{folder}/label_{batch_idx}.png")

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
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        # self.train_logger.info("Validatoin epoch {} ends, val_loss = {}".format(self.current_epoch, avg_loss))
        self.log('valid/loss', avg_loss, logger=True)
        self.log('valid_IOU', avg_iou, logger=True)
        self.log('valid_ACC', avg_acc, logger=True)
        print('============end validation==============')

    # def test_step_end(self, outputs):
    #     # for x in outputs:
    #     if outputs['iou']<0.8:
    #         self.sumresult['80'] +=1
    #     elif 0.8<outputs['iou']<0.9:
    #         self.sumresult['90'] +=1
    #     else:
    #         self.sumresult['95'] +=1

    def test_epoch_end(self, outputs):

        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.train_logger.info("Test epoch {} ends, val_loss = {}".format(self.current_epoch, avg_iou))
        self.log('valid/IOU', avg_iou, logger=True)

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.hparamss['lr'])
        # return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        # return torch.optim.Adam(self.parameters(),lr=self.learning_rate)


class mutil_train(unet_train):
    def __init__(self, hparams):
        super().__init__(hparams)

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        print('============start validation==============')
        folder = "saved_images/"
        x, y = batch
        y_hat = self(x)
        loss = self.loss.forward(y_hat, y.unsqueeze(dim=1))

        self.log("loss", loss, on_epoch=True)


        # argmax
        preds = torch.softmax(y_hat, dim=1)
        pred = preds.argmax(dim=1)


        # preds = torch.sigmoid(y_hat.squeeze())
        # pred = (preds > 0.6).float()

        iou = torchmetrics.IoU(num_classes=2, absent_score=1, reduction='none').cuda()
        validation_recall = torchmetrics.Recall(average='macro', mdmc_average='samplewise', multiclass=True,
                                                num_classes=2).cuda()
        validation_precision = torchmetrics.Precision(average='macro', mdmc_average='samplewise', multiclass=True,
                                                      num_classes=2).cuda()
        validation_ACC = torchmetrics.Accuracy().cuda()
        # pred=preds.int().unsqueeze(dim=1)
        y = y.long()
        RS_IOU = iou(pred.unsqueeze(1), y.unsqueeze(1))
        RS_recall = validation_recall(pred.unsqueeze(1), y.unsqueeze(1))
        RS_precision = validation_precision(pred.unsqueeze(1), y.unsqueeze(1))
        acc = validation_ACC(pred.unsqueeze(1), y.unsqueeze(1))

        self.log("IOU:", RS_IOU[1], prog_bar=True)
        # self.log("RS_recall:", RS_recall, prog_bar=True)
        # self.log("RS_precision:", RS_precision, prog_bar=True)
        # self.log("acc:", acc, prog_bar=True)

        torchvision.utils.save_image(
            pred.unsqueeze(dim=1).cpu(), f"{folder}/pred_{batch_idx}.png"
        )

        torchvision.utils.save_image(y.float().unsqueeze(dim=1).cpu(), f"{folder}/label_{batch_idx}.png")

        return {"loss": loss,
                "iou": RS_IOU[1],
                'acc': acc}
