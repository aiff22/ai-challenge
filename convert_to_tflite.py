# The following instructions will show you how to convert a sample SRCNN model to the TFLite format

import tensorflow as tf


def SRCNN(image_):

    # Defining the architecture of the SRCNN model

    weights = {
        'w1': tf.Variable(tf.compat.v1.random_normal([9, 9, 3, 64], stddev=1e-3), name='w1'),
        'w2': tf.Variable(tf.compat.v1.random_normal([5, 5, 64, 32], stddev=1e-3), name='w2'),
        'w3': tf.Variable(tf.compat.v1.random_normal([5, 5, 32, 3], stddev=1e-3), name='w3')
    }

    biases = {
        'b1': tf.Variable(tf.zeros([64]), name='b1'),
        'b2': tf.Variable(tf.zeros([32]), name='b2'),
        'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

    conv1 = tf.nn.relu(tf.nn.conv2d(image_, weights['w1'], strides=[1,1,1,1], padding='SAME') + biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='SAME') + biases['b2'])
    conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='SAME') + biases['b3']

    return conv3


with tf.compat.v1.Session() as sess:

    # Placeholders for input data
    # The values of the input image should lie in the interval [0, 255]
    # ------------------------------------------------------------------
    x_ = tf.compat.v1.placeholder(tf.float32, [1, 1024, 1536, 3], name="input")

    # Perform image preprocessing (e.g., normalization, scaling, etc.)
    x_norm = x_ / 255.0

    # Process the image with a sample SRCNN model
    processed = SRCNN(x_norm)

    # Scale the processed image so that its values lie in the interval [0, 255]
    output_ = tf.identity(processed * 255, name="output")

    # Load your pre-trained model
    # saver = tf.compat.v1.train.Saver()
    # saver.restore(sess, "path/to/your/saved/model")

    # In this example, we just initialize it with some random values
    sess.run(tf.compat.v1.global_variables_initializer())

    # Export your model to the TFLite format
    # Note that the "experimental_new_converter" flag is enabled by default in TensorFlow 2.2+

    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [x_], [output_])
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)

    # That is it! Your model is now saved as model.tflite file
    # You can now try to run it using the PRO mode of the AI Benchmark application:
    # https://play.google.com/store/apps/details?id=org.benchmark.demo
    # More details can be found here (Running Custom TensorFlow Lite Models):
    # http://ai-benchmark.com/news_2020_05_31_may_release.html
    # -----------------------------------------------------------------------------

