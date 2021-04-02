from tensorflow.keras.layers import Dense, Layer
from tensorflow.nn import tanh, softmax
from tensorflow import expand_dims, reduce_sum, zeros, concat
from tf.math import add


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.W3 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        '''
          query = s_t-1
          values = h_t
          coverage = coverage attention sum vector
        '''
        num_timesteps = values.shape[1]
        e_t = self.V(tanh(self.W1(query)+self.W2(values[:, 0, :])))
        score = expand_dims(e_t, axis=1)
        # attention = softmax(score, axis=1)
        attention_weights = softmax(score, axis=1)
        coverage_vector = zeros(shape=attention_weights.shape)

        for i in range(1, num_timesteps):
            sum_t = add(coverage_vector[:, i-1, :], attention_weights[:, i-1, :])
            coverage_vector = concat([coverage_vector, expand_dims(sum_t, axis=1)], axis=1)
            e_t = self.V(tanh(self.W1(query) + self.W2(values[:, i, :]) + self.W3(coverage_vector[:, i, :])))
            score = expand_dims(e_t, axis=1)
            attention = softmax(score, axis=1)
            attention_weights = concat([attention_weights, attention], axis=1)

        context_vector = attention_weights * values
        context_vector = reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights, coverage_vector
