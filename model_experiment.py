import torch
import torch.nn as nn


class Residualblock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residualblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 1x1conv来升维
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.conv3 = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.bn1(X)
        Y = self.conv1(Y)
        Y = self.bn2(Y)
        Y = self.conv2(Y)

        # 1x1conv对浅层输入的X升维
        if self.conv3:
            X = self.conv3(X)
        return self.relu(Y + X)


class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = Residualblock(in_c + out_c, out_c, use_1x1conv=True)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        if x.shape != skip.shape:
            # x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x


class UNET_res(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256]):
        super(UNET_res, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=3, stride=1, padding=1)
        self.br1 = batchnorm_relu(features[0])
        self.c12 = nn.Conv2d(features[0], features[0], kernel_size=3, padding=1)
        self.s13 = nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        for feature in features:
            self.downs.append(Residualblock(feature, feature * 2, use_1x1conv=True, stride=2))

        """ Decoder """
        for feature in reversed(features):
            self.ups.append(decoder_block(feature * 2, feature)
                            )

        # self.bottleneck = Residualblock(features[-1], features[-1] * 2,use_1x1conv=True,stride=2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.conv1(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.s13(inputs)
        skip_connections = []
        skip = x + s
        skip_connections.append(skip)
        for down in self.downs:
            skip = down(skip)
            skip_connections.append(skip)

        # x = self.bottleneck(x)
        # skip_connections = skip_connections[::-1]

        for up, t in zip(self.ups, reversed(skip_connections[:-1])):
            d = up(skip, t)
            skip = d

        return self.final_conv(skip)


backbone = 'resnet50'


class DecoderBlock(nn.Module):
    """
    U-Net中的解码模块
    采用每个模块一个stride为1的3*3卷积加一个上采样层的形式
    上采样层可使用'deconv'、'pixelshuffle', 其中pixelshuffle必须要mid_channels=4*out_channles
    定稿采用pixelshuffle
    BN_enable控制是否存在BN，定稿设置为True
    """

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1,
                              bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,

                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu2(x)
        return x

import torchvision.models as models

class Resnet_Unet(nn.Module):
    """
    定稿使用resnet50作为backbone
    BN_enable控制是否存在BN，定稿设置为True
    """

    def __init__(self, BN_enable=True, resnet_pretrain=False):
        super().__init__()
        self.BN_enable = BN_enable
        # encoder部分
        # 使用resnet34或50预定义模型，由于单通道入，因此自定义第一个conv层，同时去掉原fc层
        # 剩余网络各部分依次继承
        # 经过测试encoder取三层效果比四层更佳，因此降采样、升采样各取4次
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=resnet_pretrain)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=resnet_pretrain)
            filters = [64, 256, 512, 1024, 2048]
        self.firstconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
            )


    def forward(self, x):
        x = self.firstconv(x)

        x = self.firstbn(x)
        x = self.firstrelu(x)

        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)

        e3 = self.encoder3(e2)

        center = self.center(e3)
        d2 = self.decoder1(torch.cat([center, e2], dim=1))
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d4 = self.decoder3(torch.cat([d3, x], dim=1))

        return self.final(d4)

if __name__ == "__main__":
    inputs = torch.randn((4, 3, 256, 448))
    model = Resnet_Unet()
    y = model(inputs)
    print(y.shape)
