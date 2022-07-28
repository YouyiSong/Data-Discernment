import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, obj_num):
        super(UNet, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.down2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(128),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(128),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.down3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.down4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True),
                                    nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.InstanceNorm2d(1024),
                                    nn.LeakyReLU(0.01, inplace=True),
                                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.Dropout(0.1),
                                    nn.InstanceNorm2d(1024),
                                    nn.LeakyReLU(0.01, inplace=True),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=True),
                                    )

        self.up1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(512),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(512),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=True),
                                 )

        self.up2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(256),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(256),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=True),
                                 )

        self.up3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(128),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(128),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=True),
                                 )

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(64),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.InstanceNorm2d(64),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(64, obj_num, kernel_size=1, bias=True),
                                 nn.Softmax(dim=1)
                                 )

    def forward(self, img):
        x1 = self.down1(img)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.center(x4)
        x = self.up1(torch.cat([x, x4], dim=1))
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))
        return x



class ResConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResConv, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.InstanceNorm2d(output_dim),
                                  nn.LeakyReLU(0.01, inplace=True),
                                  nn.Conv2d(output_dim, output_dim, kernel_size=3,  stride=1, padding=1, bias=True)
                                  )

        self.skip = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, x):
        return self.conv(x) + self.skip(x)



class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpSample, self).__init__()

        self.up = nn.Sequential(nn.InstanceNorm2d(input_dim),
                                nn.LeakyReLU(0.01, inplace=True),
                                nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2, bias=True)
                                )

    def forward(self, x):
        return self.up(x)


class DownSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownSample, self).__init__()

        self.down = nn.Sequential(nn.InstanceNorm2d(input_dim),
                                  nn.LeakyReLU(0.01, inplace=True),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=True)
                                  )

    def forward(self, x):
        return self.down(x)



class ResNet(nn.Module):
    def __init__(self, obj_num):
        super(ResNet, self).__init__()
        self.conv1 = ResConv(1, 64)

        self.down1 = DownSample(64, 64)

        self.conv2 = ResConv(64, 128)

        self.down2 = DownSample(128, 128)

        self.conv3 = ResConv(128, 256)

        self.down3 = DownSample(256, 256)

        self.conv4 = ResConv(256, 512)

        self.down4 = DownSample(512, 512)

        self.conv5 = nn.Sequential(ResConv(512, 1024), nn.Dropout(0.1))

        self.up1 = UpSample(1024, 512)

        self.conv6 = ResConv(1024, 512)

        self.up2 = UpSample(512, 256)

        self.conv7 = ResConv(512, 256)

        self.up3 = UpSample(256, 128)

        self.conv8 = ResConv(256, 128)

        self.up4 = UpSample(128, 64)

        self.conv9 = ResConv(128, 64)

        self.out = nn.Sequential(nn.InstanceNorm2d(64),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(64, obj_num, kernel_size=1, bias=True),
                                 nn.Softmax(dim=1)
                                 )

    def forward(self, x):
        x1 = self.conv1(x)
        x1_d = self.down1(x1)
        x2 = self.conv2(x1_d)
        x2_d = self.down2(x2)
        x3 = self.conv3(x2_d)
        x3_d = self.down3(x3)
        x4 = self.conv4(x3_d)
        x4_d = self.down4(x4)
        x = self.conv5(x4_d)
        x = self.up1(x)
        x = self.conv6(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv7(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv8(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv9(torch.cat([x, x1], dim=1))
        x= self.out(x)

        return x



class IncepConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IncepConv, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(input_dim, int(output_dim/4), kernel_size=1, stride=1, bias=True),
                                     nn.InstanceNorm2d(int(output_dim/4)),
                                     nn.LeakyReLU(0.01, inplace=True),
                                     )

        self.branch3 = nn.Sequential(nn.Conv2d(input_dim, int(input_dim/4), kernel_size=1, stride=1, bias=True),
                                     nn.InstanceNorm2d(int(input_dim/4)),
                                     nn.LeakyReLU(0.01, inplace=True),
                                     nn.Conv2d(int(input_dim/4), int(output_dim/4), kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.InstanceNorm2d(int(output_dim/4)),
                                     nn.LeakyReLU(0.01, inplace=True),
                                     nn.Conv2d(int(output_dim/4), int(output_dim/4), kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.InstanceNorm2d(int(output_dim/4)),
                                     nn.LeakyReLU(0.01, inplace=True),
                                     )

        self.branch5 = nn.Sequential(nn.Conv2d(input_dim, int(input_dim/4), kernel_size=1, stride=1, bias=True),
                                     nn.InstanceNorm2d(int(input_dim/4)),
                                     nn.LeakyReLU(0.01, inplace=True),
                                     nn.Conv2d(int(input_dim/4), int(output_dim/4), kernel_size=5, stride=1, padding=2, bias=True),
                                     nn.InstanceNorm2d(int(output_dim/4)),
                                     nn.LeakyReLU(0.01, inplace=True),
                                     )

        self.branch_p = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(input_dim, int(output_dim/4), kernel_size=1, stride=1, bias=True),
                                      nn.InstanceNorm2d(int(output_dim/4)),
                                      nn.LeakyReLU(0.01, inplace=True),
                                      )
        self.out = nn.Sequential(nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=1, bias=True),
                                 nn.InstanceNorm2d(output_dim),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_p = self.branch_p(x)
        out = torch.cat((branch1, branch3, branch5, branch_p), dim=1)
        out =self.out(out)
        return out



class Inception(nn.Module):
    def __init__(self, obj_num):
        super(Inception, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.down2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
                                   IncepConv(64, 128)
                                   )

        self.down3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                   IncepConv(128, 256)
                                   )

        self.down4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                                   IncepConv(256, 512)
                                   )

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True),
                                    IncepConv(512, 1024),
                                    nn.Dropout(0.1),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=True)
                                    )

        self.up1 = nn.Sequential(IncepConv(1024, 512),
                                 nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=True)
                                 )

        self.up2 = nn.Sequential(IncepConv(512, 256),
                                 nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=True)
                                 )

        self.up3 = nn.Sequential(IncepConv(256, 128),
                                 nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=True)
                                 )

        self.up4 = nn.Sequential(IncepConv(128, 64),
                                 nn.Conv2d(64, obj_num, kernel_size=1, bias=True),
                                 nn.Softmax(dim=1)
                                 )

    def forward(self, img):
        x1 = self.down1(img)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.center(x4)
        x = self.up1(torch.cat([x, x4], dim=1))
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))
        return x