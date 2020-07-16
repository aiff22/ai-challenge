# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

# Note: install the latest onnx_tf library from github using the following command:
# pip install --user https://github.com/onnx/onnx-tensorflow/archive/master.zip

import torch
import torch.nn as nn

from onnx_tf.backend import prepare
import onnx

import tensorflow as tf


def conv_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU()
    )


def conv_conv_2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2), nn.ReLU()
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down_1 = conv_conv_2(3, 8)
        self.down_2 = conv_conv_2(8, 16)
        self.down_3 = conv_conv_2(16, 32)
        self.down_4 = conv_conv_2(32, 64)

        self.bottom = conv_conv(64, 64)

        self.up_1 = conv_conv(64, 32)
        self.up_2 = conv_conv(32, 16)
        self.up_3 = conv_conv(16, 8)

        self.conv_final = nn.Conv2d(8, 3, 1, padding=0)

        self.upsample_0 = torch.nn.Upsample(scale_factor=2)
        self.upsample_1 = torch.nn.Upsample(scale_factor=2)
        self.upsample_2 = torch.nn.Upsample(scale_factor=2)
        self.upsample_3 = torch.nn.Upsample(scale_factor=2)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)

        x = self.upsample_0(self.bottom(x))
        x = self.upsample_1(self.up_1(x))
        x = self.upsample_2(self.up_2(x))
        x = self.upsample_3(self.up_3(x))

        return self.conv_final(x)


if __name__ == '__main__':

    # Creating / loading pre-trained PyNET model

    model = UNet()
    model.eval()

    # Converting model to ONNX

    for _ in model.modules():
        _.training = False

    sample_input = torch.randn(1, 3, 1024, 1536)
    input_nodes = ['input']
    output_nodes = ['output']

    torch.onnx.export(model, sample_input, "model.onnx", input_names=input_nodes, output_names=output_nodes)

    # Converting model to Tensorflow

    onnx_model = onnx.load("model.onnx")
    output = prepare(onnx_model)
    output.export_graph("model.pb")

    # Use the following Python script to convert your model to NHWC format:
    #
    # https://github.com/paulbauriegel/tensorflow-tools/blob/master/convert-model-to-NHWC.py
    #
    # Running this will result in model_toco.pb file
    #
    # Note that this step is absolutely necessary since the TFLite is not supporting NCHW format!

    # Converting model to TFLite

    g = tf.Graph()
    with tf.compat.v1.Session() as sess:

        with tf.compat.v1.gfile.FastGFile("model_toco.pb", 'rb') as f:

            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

            x_ = sess.graph.get_tensor_by_name('input:0')
            output_ = sess.graph.get_tensor_by_name('output:0')

        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [x_], [output_])
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()
        open("model.tflite", "wb").write(tflite_model)
