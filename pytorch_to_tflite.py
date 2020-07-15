# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import torch
import torch.nn as nn
import tensorflow as tf

from pytorch2keras import pytorch_to_keras


def conv_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU()
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down_1 = conv_conv(3, 8)
        self.down_2 = conv_conv(8, 16)
        self.down_3 = conv_conv(16, 32)
        self.down_4 = conv_conv(32, 64)

        self.bottom = conv_conv(64, 64)

        self.up_1 = conv_conv(64, 32)
        self.up_2 = conv_conv(32, 16)
        self.up_3 = conv_conv(16, 8)

        self.conv_final = nn.Conv2d(8, 3, 1, padding=0)

        self.upsample_0 = torch.nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2)
        self.upsample_1 = torch.nn.ConvTranspose2d(32, 32, 4, padding=1, stride=2)
        self.upsample_2 = torch.nn.ConvTranspose2d(16, 16, 4, padding=1, stride=2)
        self.upsample_3 = torch.nn.ConvTranspose2d(8, 8, 4, padding=1, stride=2)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.max_pool(self.down_1(x))
        x = self.max_pool(self.down_2(x))
        x = self.max_pool(self.down_3(x))
        x = self.max_pool(self.down_4(x))

        x = self.upsample_0(self.bottom(x))
        x = self.upsample_1(self.up_1(x))
        x = self.upsample_2(self.up_2(x))
        x = self.upsample_3(self.up_3(x))

        return self.conv_final(x)


if __name__ == '__main__':

    # Creating / loading pre-trained PyNET model

    model = UNet()
    model.eval()

    # Converting model to Keras

    for _ in model.modules():
        _.training = False

    sample_input = torch.randn(1, 3, 1024, 1536)
    input_nodes = ['input']
    output_nodes = ['output']

    k_model = pytorch_to_keras(model, sample_input, change_ordering=True, verbose=True)
    k_model.save("model.h5")

    # Converting model to TFLite

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("model.h5")
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)

