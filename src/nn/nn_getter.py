import src.nn.config as config
import tensorflow as tf
from src.data.getter.getter import DataGetter
import numpy as np


def build_net(norm, on_train):
    def add_layer(inputs, input_size, output_size, is_on_train, activation_function=tf.nn.relu, is_norm=True):
        weights = tf.Variable(tf.random_normal([input_size, output_size], mean=0., stddev=1.), dtype=tf.float32)
        biases = tf.Variable(tf.zeros([1, output_size]) + 0.1, dtype=tf.float32)

        wx_plus_b = tf.matmul(inputs, weights) + biases
        wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob=config.KEEP_PROB)

        # normalize fully connected product
        if is_norm:
            # Batch Normalize
            fc_mean_layer, fc_var_layer = tf.nn.moments(
                wx_plus_b,
                axes=[0],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([output_size]))
            shift = tf.Variable(tf.zeros([output_size]))
            epsilon = 0.001

            # apply moving average for mean and var when train on batch
            ema_layer = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema_layer.apply([fc_mean_layer, fc_var_layer])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean_layer), tf.identity(fc_var_layer)

            mean_layer, var_layer = None, None
            if is_on_train:
                mean_layer, var_layer = mean_var_with_update()
            else:
                mean_layer, var_layer = lambda: (ema_layer.average(fc_mean_layer), ema_layer.average(fc_var_layer))

            wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean_layer, var_layer, shift, scale, epsilon)

        # activation
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)

        return outputs

    xs = tf.placeholder(dtype=tf.float32, shape=[None, config.N_INPUT])
    ys = tf.placeholder(dtype=tf.float32, shape=[None, config.N_LABEL])
    # add normalization on input layer
    data_getter = DataGetter()
    non_test_data = data_getter.get_non_test_records()
    feature_array = [data.features for data in non_test_data]
    mean_input = np.mean(feature_array, axis=0)
    stddev_input = np.std(feature_array, axis=0)
    print("input_mean = {}".format(mean_input))
    print("stddev_input = {}".format(stddev_input))
    xs = (xs - mean_input) / stddev_input

    # record inputs for every layer
    layers_inputs = [xs]

    # build hidden layers
    for l_n in range(config.N_HIDDEN_LAYER):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(
            layer_input,    # input
            in_size,        # input size
            config.N_HIDDEN_UNIT,  # output size
            on_train,       # is training
            config.ACTIVATION,     # activation function
            norm,           # normalize before activation
        )
        layers_inputs.append(output)    # add output for next run

    # build output layer
    prediction = add_layer(layers_inputs[-1], config.N_HIDDEN_UNIT, config.N_LABEL, on_train, activation_function=None)

    global_step = tf.Variable(0, trainable=False)
    return global_step, xs, prediction, ys
