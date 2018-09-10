import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

class PyramidPool(nn.Module):

    def __init__(self, in_features, out_features, pool_size):
        super(PyramidPool, self).__init__()

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features, momentum=.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()
        output = F.upsample(self.features(x), size[2:], mode='bilinear')
        return output


class PSPNet(nn.Module):

    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        init_net = models.resnet101(pretrained=True)
        self.resnet = init_net

        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)

        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print("initialize ", m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                print("initialize ", m)

    def forward(self, x):
        count = 0

        size = x.size()
        x = self.resnet(x)

        x = x[0]

        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))

        return F.upsample_bilinear(x, size[2:])

if __name__ == '__main__':
        # test
    net = PSPNet(1).cuda()
    pdb.set_trace()
    net(torch.ones([1, 1, 101, 101]).cuda())
    pass