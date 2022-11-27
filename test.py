import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 1x1conv来升维
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 1x1conv对浅层输入的X升维
        if self.conv3:
            X = self.conv3(X)
        return self.relu(Y + X)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock3, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=1, res=True):
        super(AttentionUNet, self).__init__()
        self.res = res
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        if res:
            nr = [3, 4, 6, 3]
            self.Conv0 = nn.Conv2d(img_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


            self.Conv1 = self._make_layer(64, 64, blocks=nr[0], stride=1)
            self.Conv2 = self._make_layer(64, 128, blocks=nr[1], stride=2)
            self.Conv3 = self._make_layer(128, 256, blocks=nr[2], stride=2)
            self.Conv4 = self._make_layer(256, 512, blocks=nr[3], stride=2)

            self.Up1 = UpConv(64, 64)
            self.Att1 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
            self.UpConv1 = ConvBlock(128, 64)
            self.Up0 = UpConv(64, 64)


        else:
            self.Conv1 = ConvBlock(img_ch, 64)
            self.Conv2 = ConvBlock(64, 128)
            self.Conv3 = ConvBlock3(128, 256)
            self.Conv4 = ConvBlock3(256, 512)
            self.Conv5 = ConvBlock3(512, 512)
            self.Up5 = UpConv(512, 512)
            self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
            self.UpConv5 = ConvBlock(1024, 512)



        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def _make_layer(
            self,
            inplanes: int,
            planes: int,
            blocks: int,
            stride: int = 1,
    ) -> nn.Sequential:

        layers = []
        layers.append(
            Residual(
                inplanes, planes, use_1x1conv=True, stride=stride)
        )
        for _ in range(1, blocks):
            layers.append(
                Residual(
                    planes, planes, use_1x1conv=True)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        if self.res:
            e0 = self.Conv0(x)
            e0 = self.bn1(e0)
            e0 = self.relu(e0)

            e1 = self.MaxPool(e0)
            e1 = self.Conv1(e1)
            e2 = self.Conv2(e1)
            e3 = self.Conv3(e2)
            e4 = self.Conv4(e3)
            d4 = self.Up4(e4)


        else:
            e1 = self.Conv1(x)

            e2 = self.MaxPool(e1)
            e2 = self.Conv2(e2)

            e3 = self.MaxPool(e2)
            e3 = self.Conv3(e3)

            e4 = self.MaxPool(e3)
            e4 = self.Conv4(e4)

            e5 = self.MaxPool(e4)
            e5 = self.Conv5(e5)

            d5 = self.Up5(e5)
            s4 = self.Att5(gate=d5, skip_connection=e4)
            d5 = torch.cat((s4, d5), dim=1)  # concatenate attention-weighted skip connection with previous layer output
            d5 = self.UpConv5(d5)
            d4 = self.Up4(d5)

        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        if self.res:
            d1 = self.Up1(d2)
            s0 = self.Att1(gate=d1, skip_connection=e0)
            d1 = torch.cat((s0, d1), dim=1)
            d1 = self.UpConv1(d1)

            d1 = self.Up0(d1)

            out = self.Conv(d1)
        else:
            out = self.Conv(d2)


        return out


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2]
        z = self.relu(z)
        z = self.Conv_Excitation(z)  # shape: [bs, c]
        z = self.norm(z)

        return U * z.expand_as(U)


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        print('U_sse', U_sse.size())
        U_cse = self.cSE(U)
        print('U_cse', U_cse.size())

        return U_cse + U_sse


def get_model_parametersum(model):
    total = sum(p.numel() for p in model.parameters())

    print("Total params: %.2fM" % (total / 1e6))


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 480, 640
    in_tensor = torch.ones(bs, c, h, w)

    model_vgg = smp.Unet(encoder_name='vgg16_bn',
                         # encoder_depth=4,
                         # decoder_channels=[512,256, 128, 64,32],
                         in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                         classes=3,  # model output channels (number of classes in your dataset)
                         # decoder_attention_type='scse'
                         ).cuda()

    model_res = smp.Unet(encoder_name='resnet34',
                         # encoder_depth=4,
                         # decoder_channels=[512,256, 128, 64,32],
                         in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                         classes=3,  # model output channels (number of classes in your dataset)
                         # decoder_attention_type='scse'
                         ).cuda()
    # summary(model_res, (3, 480, 640))

    model_pp = smp.UnetPlusPlus(
        # encoder_depth=4,
        # decoder_channels=[512,256, 128, 64,32],
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
        # decoder_attention_type='scse'
    ).cuda()

    # A = AttentionUNet(res=True)
    # A2 = AttentionUNet(res=False)
    # print("in shape:", in_tensor.shape)
    # out_tensor = A(in_tensor)
    # print("out shape:", out_tensor.shape)

    # get_model_parametersum(model_vgg)
    get_model_parametersum(model_res)
    # get_model_parametersum(model_pp)
    # get_model_parametersum(A)
    # get_model_parametersum(A2)
    summary(model_res, (3, 480, 640))
