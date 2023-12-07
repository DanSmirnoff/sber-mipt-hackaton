import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels_=1, out_channels_=1):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = self.encode_seq_(in_channels_, 16)
        self.pool0 = nn.MaxPool2d(kernel_size=2)    # 512 -> 256

        self.enc_conv1 = self.encode_seq_(16, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)    # 256 -> 128

        self.enc_conv2 = self.encode_seq_(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)    # 128 -> 64

        self.enc_conv3 = self.encode_seq_(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)    # 64 -> 32

        # bottleneck
        self.bottleneck_conv = self.encode_seq_(128, 256)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2)    # 32 -> 64
        self.dec_conv0 = self.decode_seq_(256+128, 128)

        self.upsample1 = nn.Upsample(scale_factor=2)    # 64 -> 128
        self.dec_conv1 = self.decode_seq_(128+64, 64)

        self.upsample2 = nn.Upsample(scale_factor=2)    # 128 -> 256
        self.dec_conv2 = self.decode_seq_(64+32, 32)

        self.upsample3 = nn.Upsample(scale_factor=2)    # 256 -> 512
        self.dec_conv3 = self.decode_seq_(32+16, 16)

        self.output_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        p0 = self.pool0(e0)

        e1 = self.enc_conv1(p0)
        p1 = self.pool1(e1)

        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)

        # bottleneck
        b = self.bottleneck_conv(p3)

        # decoder
        d0 = self.upsample0(b)
        d0 = torch.cat((d0, e3), dim=1)
        d0 = self.dec_conv0(d0)

        d1 = self.upsample1(d0)
        d1 = torch.cat((d1, e2), dim=1)
        d1 = self.dec_conv1(d1)

        d2 = self.upsample2(d1)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec_conv2(d2)

        d3 = self.upsample3(d2)
        d3 = torch.cat((d3, e0), dim=1)
        d3 = self.dec_conv3(d3)

        d3 = self.output_conv(d3)    # no activation

        return d3

    def encode_seq_(self, in_channels_, out_channels_):
        seq_ = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels_),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels_, out_channels=out_channels_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels_),
            nn.ReLU()
        )

        return seq_

    def decode_seq_(self, in_channels_, out_channels_):
        seq_ = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels_),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels_, out_channels=out_channels_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels_),
            nn.ReLU()
        )

        return seq_
