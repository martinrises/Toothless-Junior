import src.nn.config as config
import tensorflow as tf
from src.data.getter.getter import DataGetter
import numpy as np


def build_net():
    def add_layer(inputs, input_size, output_size, activation_function=tf.nn.relu):
        weights = tf.Variable(tf.random_normal([input_size, output_size], mean=0., stddev=1.), dtype=tf.float32)
        biases = tf.Variable(tf.zeros([1, output_size]) + 0.1, dtype=tf.float32)

        wx_plus_b = tf.matmul(inputs, weights) + biases
        wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob=config.KEEP_PROB)

        # activation
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)

        return outputs

    xs = tf.placeholder(dtype=tf.float32, shape=[None, config.N_INPUT])
    ys = tf.placeholder(dtype=tf.float32, shape=[None, config.N_LABEL])

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
            config.ACTIVATION,     # activation function
        )
        layers_inputs.append(output)    # add output for next run

    # build output layer
    prediction = add_layer(layers_inputs[-1], config.N_HIDDEN_UNIT, config.N_LABEL, on_train, activation_function=None)

    global_step = tf.Variable(0, trainable=False)
    return global_step, xs, prediction, ys
