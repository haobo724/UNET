import os

import numpy as np

import glob

import torchmetrics

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from model import UNET, UNET_S,UNet_PP
from utils import (
    get_loaders,

)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser
import PIL.Image as Image

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 50
NUM_WORKERS = 4
# IMAGE_HEIGHT = 274  # 1096 originally  0.25
# IMAGE_WIDTH = 484  # 1936 originally
IMAGE_HEIGHT = 480  # 1096 originally  0.25
IMAGE_WIDTH = 640  # 1936 originally
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/all_images/"
TRAIN_MASK_DIR = "data/all_masks/"
VAL_IMG_DIR = "data/all_images/"
VAL_MASK_DIR = "data/all_masks/"


def add_training_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--data_folder', nargs='+', type=str)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--mode_size', type=int, default=64)

    return parser


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
        self.sumresult = {}
        self.sumresult.setdefault('80', 0)
        self.sumresult.setdefault('90', 0)
        self.sumresult.setdefault('95', 0)

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

        self.log("loss", loss, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        folder = "saved_images/"
        x, y = batch
        y_hat = self(x)
        loss = self.loss.forward(y_hat, y.unsqueeze(dim=1))

        self.log("loss", loss, on_epoch=True)

        preds = torch.sigmoid(y_hat.squeeze())
        pred = (preds > 0.6).float()

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
        self.log("RS_recall:", RS_recall, prog_bar=True)
        self.log("RS_precision:", RS_precision, prog_bar=True)
        self.log("acc:", acc, prog_bar=True)

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
        # self.train_logger.info("Training epoch {} ends".format(self.current_epoch))
        self.log('train/loss', avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        # self.train_logger.info("Validatoin epoch {} ends, val_loss = {}".format(self.current_epoch, avg_loss))
        self.log('valid/loss', avg_loss, logger=True)
        self.log('valid_IOU', avg_iou, logger=True)
        self.log('valid_ACC', avg_acc, logger=True)

    # def test_step_end(self, outputs):
    #     # for x in outputs:
    #     if outputs['iou']<0.8:
    #         self.sumresult['80'] +=1
    #     elif 0.8<outputs['iou']<0.9:
    #         self.sumresult['90'] +=1
    #     else:
    #         self.sumresult['95'] +=1

    def test_epoch_end(self, outputs):
        for x in outputs:
            if x['iou'] < 0.8:
                self.sumresult['80'] += 1
            elif 0.8 < x['iou'] < 0.9:
                self.sumresult['90'] += 1
            else:
                self.sumresult['95'] += 1
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.train_logger.info("Test epoch {} ends, val_loss = {}".format(self.current_epoch, avg_iou))
        self.log('valid/IOU', avg_iou, logger=True)
        self.log('sumresult', self.sumresult, logger=True)

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.hparamss['lr'])
        # return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        # return torch.optim.Adam(self.parameters(),lr=self.learning_rate)


def main():
    pl.seed_everything(1234)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ColorJitter(brightness=0.3, hue=0.3, p=0.4),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.2),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    parser = ArgumentParser()
    parser = add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = unet_train.add_model_specific_args(parser)
    args = parser.parse_args()

    model = unet_train(hparams=vars(args)).cuda()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    if args.mode_size == 32:
        name = 'S'
    elif args.mode_size == 16:
        name = 'XS'
    else:
        name = 'M'
    ckpt_callback = ModelCheckpoint(
        monitor='valid_IOU',
        save_top_k=2,
        mode='max',
        filename=f'{name}' + '{epoch:02d}-{valid_IOU:02f}'

    )
    logger = TensorBoardLogger(save_dir=os.path.join('.', 'lightning_logs'), name='my_model')
    trainer = pl.Trainer.from_argparse_args(args, check_val_every_n_epoch=3, callbacks=[ckpt_callback], logger=logger)


    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}')

    # make the direcrory for the checkpoints
    if not os.path.exists(os.path.join('.', 'lightning_logs', f'version_{trainer.logger.version}')):
        os.makedirs(os.path.join('.', 'lightning_logs', f'version_{trainer.logger.version}'))

    trainer.fit(model, train_loader, val_loader)
    # trainer.save_checkpoint(
    #     os.path.join('.', 'lightning_logs', f'version_{trainer.logger.version}', 'final.ckpt'))
    print('THE END')


if __name__ == "__main__":
    # modelslist = []
    # for root, dirs, files in os.walk(r".\lightning_logs"):
    #     for file in files:
    #         if file.endswith('.ckpt'):
    #             modelslist.append(os.path.join(root, file))
    # print(modelslist)
    # print(modelslist[-3])
    #
    # infer(modelslist[-3], './testdata')

    main()
