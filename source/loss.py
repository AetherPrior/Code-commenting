import tensorflow as tf


def coverage_loss(targets, predictions, attention_weights, coverage_vector, coverage_lambda=0.5):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM)
    covloss = tf.reduce_sum(tf.minimum(
        coverage_vector, attention_weights), axis=1)
    return loss_object(targets, predictions) + coverage_lambda*covloss
