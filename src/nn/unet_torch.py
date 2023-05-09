import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, resnet_style=False):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resnet_style = resnet_style

    def forward(self, x):
        encoder = self.conv_block(x)
        if self.resnet_style:
            encoder = torch.cat([encoder, x], dim=1)
        encoder_pool = self.pool(encoder)
        return encoder_pool, encoder

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, concat_channels, out_channels, activation=nn.ReLU):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels + concat_channels)
        self.activation = activation()
        self.conv1 = nn.Conv2d(out_channels + concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, concat_tensor):
        x = self.upconv(x)
        if concat_tensor is not None:
            x = torch.cat([concat_tensor, x], dim=1)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.activation(x)
        return x
