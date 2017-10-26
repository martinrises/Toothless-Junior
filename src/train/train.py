import tensorflow as tf
import numpy as np
import src.nn.nn_getter as nn_getter
import src.nn.config as config
from src.data.getter.getter import DataGetter
import random


def get_random_segment(records, batch_size=config.BATCH_SIZE):
    index = random.choice(range(len(records) - batch_size + 1))
    return records[index: index + batch_size]


def train():

    # conduct neural network
    global_step, inputs, outputs, targets = nn_getter.build_net(False, True)

    # training fields
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=outputs))
    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE).minimize(loss, global_step=global_step)

    # summary fields
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope("summary"):
        tf.summary.scalar("train_loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()

    # checkpoints
    saver = tf.train.Saver()

    # data
    data_getter = DataGetter()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(config.TRAIN_SUMMARY_DIR, sess.graph)
        cv_writer = tf.summary.FileWriter(config.CV_SUMMARY_DIR, sess.graph)
        test_writer = tf.summary.FileWriter(config.TEST_SUMMARY_DIR, sess.graph)

        ckpt = tf.train.get_checkpoint_state(config.CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(config.MAX_EPOCH):
            train_records = data_getter.get_random_training_data()
            iter_size = len(train_records) // config.BATCH_SIZE

            for iteration in range(iter_size):
                batch_records = train_records[iteration * config.BATCH_SIZE: (iteration + 1) * config.BATCH_SIZE]

                input_data = np.reshape(DataGetter.get_features(batch_records), [config.BATCH_SIZE, config.N_INPUT])
                target_data = np.reshape(DataGetter.get_labels(batch_records), [config.BATCH_SIZE, config.N_LABEL])

                _ = sess.run(optimizer, feed_dict={inputs: input_data, targets: target_data})
                if iteration % 10 == 0:
                    data_loss, train_summary = sess.run([loss, merged_summary], feed_dict={inputs: input_data, targets: target_data})
                    train_writer.add_summary(train_summary, global_step=(global_step.eval(sess)))
                    print("epoch #{} iteration = {}, loss = {}".format((global_step.eval(sess) // iter_size), (global_step.eval(sess)), data_loss))

                    batch_cv_records = get_random_segment(data_getter.get_random_cv_data())
                    cv_summary = sess.run(merged_summary, feed_dict={
                        inputs: np.reshape(data_getter.get_features(batch_cv_records), [config.BATCH_SIZE, config.N_INPUT]),
                        targets: np.reshape(data_getter.get_labels(batch_cv_records), [config.BATCH_SIZE, config.N_LABEL])})
                    cv_writer.add_summary(cv_summary, global_step=(global_step.eval(sess)))

                    batch_test_records = get_random_segment(data_getter.get_test_data())
                    test_summary = sess.run(merged_summary, feed_dict={
                        inputs: np.reshape(data_getter.get_features(batch_test_records), [config.BATCH_SIZE, config.N_INPUT]),
                        targets: np.reshape(data_getter.get_labels(batch_test_records), [config.BATCH_SIZE, config.N_LABEL])})
                    test_writer.add_summary(test_summary, global_step=(global_step.eval(sess)))

            if ((global_step.eval(sess) // iter_size) + 1) % 5 == 0:
                saver.save(sess, config.CKPT_DIR, global_step=(global_step.eval(sess) // iter_size))


train()
