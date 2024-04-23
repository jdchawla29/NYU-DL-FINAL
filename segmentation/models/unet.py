import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
      super().__init__()
      self.ConvBlock = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Dropout(0.2),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Dropout(0.2)
      )

    def forward(self, x):
        return self.ConvBlock(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
      super().__init__()
      self.encoder = nn.Sequential(
          ConvBlock(in_channels, out_channels),
          nn.MaxPool2d(2)
      )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_features):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.start = (ConvBlock(in_channels, 64))
        self.encoder1 = Encoder(64, 128)
        self.encoder2 = Encoder(128, 256)
        self.encoder3 = Encoder(256, 512)
        self.encoder4 = Encoder(512, 1024)
        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
      x1 = self.start(x)
      x2 = self.encoder1(x1)
      x3 = self.encoder2(x2)
      x4 = self.encoder3(x3)
      x5 = self.encoder4(x4)
      d1 = self.decoder1(x5, x4)
      d2 = self.decoder2(d1, x3)
      d3 = self.decoder3(d2, x2)
      d4 = self.decoder4(d3, x1)
      output = self.output_conv(d4)
      return output