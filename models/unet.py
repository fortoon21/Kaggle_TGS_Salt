import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import pdb

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch,):
        super(double_conv, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv=double_conv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(x)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv=nn.Conv2d(in_ch, out_ch,1)
    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv=nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.mpconv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else :
            self.up=nn.ConvTranspose2d(in_ch//2, in_ch//2,2, stride=2)
        self.conv=double_conv(in_ch, out_ch)

    def forward(self, x1,x2):
        x1=self.up(x1)
        diffX=x1.size()[2]-x2.size()[2]
        diffY=x1.size()[3]-x2.size()[3]
        x2=F.pad(x2, (diffX//2, ceil(diffX/2), diffY//2, ceil(diffY/2)))
        x=torch.cat([x2,x1], dim=1)
        x=self.conv(x)
        return x


class Unet(nn.Module):
	def __init__(self,in_dim,out_dim,num_filter):
		super(Unet,self).__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim
		self.num_filter = num_filter



		self.inc=inconv(in_dim, self.num_filter)
		self.down1 = down(self.num_filter,self.num_filter*2)
		self.down2 = down(self.num_filter*2,self.num_filter*4)
		self.down3 = down(self.num_filter*4,self.num_filter*8)
		self.down4= down(self.num_filter*8,self.num_filter*8)

		self.up1 = up(self.num_filter*16,self.num_filter*4)
		self.up2 = up(self.num_filter*8,self.num_filter*2)
		self.up3 = up(self.num_filter*4,self.num_filter*1)
		self.up4 = up(self.num_filter*2,self.num_filter*1)

		self.out=outconv(self.num_filter, out_dim)



	def forward(self,x):
		x1=self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)

		x = self.up1(x5,x4)
		x = self.up2(x,x3)
		x = self.up3(x,x2)
		x = self.up4(x,x1)

		out = self.out(x)

		return out

if __name__ == '__main__':
    # test
    net = Unet(1,1,64).cuda()
    pdb.set_trace()
    net(torch.ones([1, 1, 101, 101]).cuda())
    pass