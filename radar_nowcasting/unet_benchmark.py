"""
Benchmark radar nowcast model.

U-Net model that predicts future frames of VIL from previous frames of VIL.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.unet_torch import ConvBlock,EncoderBlock,DecoderBlock


# Approximate mean, std of training data
MEAN=33.44
SCALE=47.54

#code for creating model in pytorch
class UNet(nn.Module):
    def __init__(self, input_shape=(13, 384, 384), start_neurons=32, num_outputs=12, activation=nn.ReLU, mean=MEAN, scale=SCALE):
        super(UNet, self).__init__()

        self.mean = mean
        self.scale = scale

        self.encoder_block0 = EncoderBlock(input_shape[0], start_neurons, activation=activation)
        self.encoder_block1 = EncoderBlock(start_neurons, start_neurons * 2, activation=activation)
        self.encoder_block2 = EncoderBlock(start_neurons * 2, start_neurons * 4, activation=activation)
        self.encoder_block3 = EncoderBlock(start_neurons * 4, start_neurons * 8, activation=activation)

        self.center = ConvBlock(start_neurons * 8, start_neurons * 32)

        self.decoder_block3 = DecoderBlock(start_neurons * 32, start_neurons * 8, start_neurons * 8, activation=activation)
        self.decoder_block2 = DecoderBlock(start_neurons * 8, start_neurons * 4, start_neurons * 4, activation=activation)
        self.decoder_block1 = DecoderBlock(start_neurons * 4, start_neurons * 2, start_neurons * 2, activation=activation)
        self.decoder_block0 = DecoderBlock(start_neurons * 2, start_neurons, start_neurons, activation=activation)

        self.output_conv = nn.Conv2d(start_neurons, num_outputs, kernel_size=1, padding=0)

    def forward(self, x):
        # Normalize inputs
        x = (x - self.mean) / self.scale

        encoder0_pool, encoder0 = self.encoder_block0(x)
        encoder1_pool, encoder1 = self.encoder_block1(encoder0_pool)
        encoder2_pool, encoder2 = self.encoder_block2(encoder1_pool)
        encoder3_pool, encoder3 = self.encoder_block3(encoder2_pool)

        center = self.center(encoder3_pool)

        decoder3 = self.decoder_block3(center, encoder3)
        decoder2 = self.decoder_block2(decoder3, encoder2)
        decoder1 = self.decoder_block1(decoder2, encoder1)
        decoder0 = self.decoder_block0(decoder1, encoder0)

        outputs = self.output_conv(decoder0)
        return outputs


def nowcast_mse(y_true,y_pred):
    """ 
    MSE loss normalized by SCALE*SCALE
    """
    return mean_squared_error(y_true,y_pred)/(SCALE*SCALE)


def nowcast_mae(y_true,y_pred):
    """
    MAE normalized by SCALE
    """
    return mean_absolute_error(y_true,y_pred)/(SCALE)



