import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class UNET_S(UNET):
    def __init__(
            self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet_PP(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        # nb_filter = [32, 64, 128, 256,512]
        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()


        for channel in nb_filter:
            self.downs.append(VGGBlock(input_channels, channel,channel))
            input_channels = channel
        # self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        # self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        # self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        # self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        nb_filter_revers=list(reversed(nb_filter))[:-1]
        for channel in nb_filter_revers:
            self.ups.append(VGGBlock( channel+channel//2, channel//2, channel//2 ))
        #
        # self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        # self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        # self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        # self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x = input
        i =0
        x_list=[]
        for down in self.downs:
            if i >0:
                x= down(self.pool(x))
                x_list.append(x)
            else:
                x=down(x)
                x_list.append(x)
            i+=1

        # x0_0 = self.conv0_0(input)
        # x1_0 = self.conv1_0(self.pool(x0_0))
        # x2_0 = self.conv2_0(self.pool(x1_0))
        # x3_0 = self.conv3_0(self.pool(x2_0))
        # x4_0 = self.conv4_0(self.pool(x3_0))
        xl_0=x_list[-1]
        x_list_revers=list(reversed(x_list))[1:]

        for up ,x_0 in zip(self.ups,x_list_revers):
            x_1 = self.up(xl_0)
            if x_0.shape != x_1.shape:
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x_1 = torch.nn.functional.interpolate(x_1, size=x_0.shape[2:])
            x_out = up(torch.cat([x_0, x_1], 1))

            xl_0=x_out
        #
        # x3_1 = self.up(x4_0)
        #
        # x3_1 = self.conv3_1(torch.cat([x3_0, x3_1], 1))
        #
        # x2_2 = self.up(x3_1)
        #
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_2], 1))
        # x1_3 = self.up(x2_2)
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_3], 1))
        # x0_4 = self.up(x1_3)
        #
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_4], 1))

        # output = self.final(x0_4)
        output = self.final(xl_0)
        return output


def test():
    x = torch.randn((3, 1, 480, 640))
    # model = UNET(in_channels=1, out_channels=1)
    model = UNet_PP(num_classes=3, input_channels=1)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
