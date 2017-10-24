import tensorflow as tf
import numpy as np
import src.nn.nn_getter as nn_getter
import src.nn.config as config
from src.data.getter.getter import DataGetter
from src.test.market.market_event_trigger import MarketEventTrigger


def test():

    # conduct neural network
    global_step, inputs, outputs, targets = nn_getter.build_net(True, True)

    # checkpoints
    saver = tf.train.Saver()

    # data
    data_getter = DataGetter()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(config.CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('no checkpoint')

        trigger = MarketEventTrigger()
        # get test data
        test_data_list = data_getter.get_test_data_list()
        for test_data_set in test_data_list:
            if len(test_data_set) == 0:
                continue

            predictions = sess.run(outputs, feed_dict={
                inputs: np.reshape(data_getter.get_features(test_data_set), [len(test_data_set), config.N_INPUT]),
                targets: np.reshape(data_getter.get_labels(test_data_set), [len(test_data_set), config.N_LABEL])
            })

            trigger.trigger_market_event(predictions, test_data_set)


test()
