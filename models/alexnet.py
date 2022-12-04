import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, drop_out=0.5):
        super(AlexNet, self).__init__()
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(48, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        dropout1 = F.dropout(conv1, p=self.drop_out, training=self.training)
        pool1 = self.pool1(dropout1)
        pool1s = torch.split(pool1, 48, dim=1)
        conv2s = [self.conv2(pool1s[i]) for i in range(len(pool1s))]
        conv2 = torch.cat(conv2s, dim=1)
        dropout2 = F.dropout(conv2, p=self.drop_out, training=self.training)
        pool2 = self.pool2(dropout2)
        return conv1, conv2, pool2
        

class SiameseAlexNet(nn.Module):
    def __init__(self, drop_out=0.5):
        super(SiameseAlexNet, self).__init__()
        self.ref_branch = AlexNet(drop_out)
        self.dis_branch = AlexNet(drop_out)
        self.deconv1 = nn.Conv2d(1025, 512, kernel_size=5, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(1024, 256, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(256, 96, kernel_size=11, stride=1, padding=5)
        self.deconv4 = nn.Conv2d(192, 96, kernel_size=11, stride=1, padding=5)
        self.deconv5 = nn.Conv2d(96, 1, kernel_size=11, stride=1, padding=5)

    def forward(self, ref, dis, ppd):
        ref_conv1, ref_conv2, ref_pool2 = self.ref_branch(ref)
        _, _, dis_pool2 = self.dis_branch(dis)
        ppd = ppd.expand(-1, -1, ref_pool2.shape[2], ref_pool2.shape[3])
        concat1 = torch.cat([dis_pool2, ref_pool2, ppd], dim=1)
        upsample1 = F.interpolate(concat1, size=ref_conv2.shape[2:])
        deconv1 = F.relu(self.deconv1(upsample1))
        concat2 = torch.cat([deconv1, ref_conv2], dim=1)
        deconv2 = F.relu(self.deconv2(concat2))
        upsample2 = F.interpolate(deconv2, size=ref_conv1.shape[2:])
        deconv3 = F.relu(self.deconv3(upsample2))
        concat3 = torch.cat([deconv3, ref_conv1], dim=1)
        deconv4 = F.relu(self.deconv4(concat3))
        upsample3 = F.interpolate(deconv4, size=ref.shape[2:])
        deconv5 = torch.sigmoid(self.deconv5(upsample3))
        return deconv5 * 255


if __name__ == '__main__':
    model = SiameseAlexNet()
    ref = torch.randn(7, 3, 48, 48)
    dis = torch.randn(7, 3, 48, 48)
    ppd = torch.randn(7)
    out = model(ref, dis, ppd)
    print(out.shape)
