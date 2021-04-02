import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        '''
          query = s_t-1
          values = h_t
          coverage = coverage attention sum vector
        '''
        num_timesteps = values.shape[1]
        score = tf.expand_dims(
            self.V(tf.nn.tanh(self.W1(query)+self.W2(values[:, 0, :]))), axis=1)

        attention = tf.nn.softmax(score, axis=1)
        attention_weights = tf.nn.softmax(score, axis=1)

        coverage_vector = tf.zeros(shape=attention_weights.shape)

        for i in range(1, num_timesteps):
            coverage_vector = tf.concat([coverage_vector, tf.expand_dims(tf.math.add(
                coverage_vector[:, i-1, :], attention_weights[:, i-1, :]), axis=1)], axis=1)
            score = tf.expand_dims(self.V(tf.nn.tanh(self.W1(
                query) + self.W2(values[:, i, :]) + self.W3(coverage_vector[:, i, :]))), axis=1)
            attention = tf.nn.softmax(score, axis=1)
            attention_weights = tf.concat(
                [attention_weights, attention], axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights, coverage_vector
