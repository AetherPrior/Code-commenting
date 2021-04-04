from tensorflow.keras.layers import Dense, Layer
from tensorflow.nn import tanh, softmax
from tensorflow import expand_dims, reduce_sum, zeros, concat
from tf.math import add


class BahdanauAttention(Layer):
    '''
    Bahdanau attention according to the pointer-generator paper:
    code can be found in: https://github.com/abisee/pointer-generator
    '''

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.W3 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        '''
          query = s_t
          values = h_t
          coverage = coverage attention sum vector
        '''

        # INITIALIZATION

        num_timesteps = values.shape[1]

        e_t = self.V(tanh(self.W1(query)+self.W2(values[:, 0, :])))
        score = expand_dims(e_t, axis=1)
        # attention = softmax(score, axis=1)

        attention_weights = softmax(score, axis=1)
        coverage_vector = zeros(shape=attention_weights.shape)

        for t in range(1, num_timesteps):
            # c_t = c_t-1 + a_t-1
            sum_t = add(coverage_vector[:, t-1, :],
                        attention_weights[:, t-1, :])

            # caching all coverage vectors over all timesteps
            coverage_vectors = concat(
                [coverage_vectors, expand_dims(sum_t, axis=1)], axis=1)

            # e_t = v_t(tanh(W_h*hi + W_s*s_t + w_c*c_t + b_attn))
            # we don't have a separate bias
            e_t = self.V(tanh(
                self.W1(query) + self.W2(values[:, t, :]) + self.W3(coverage_vectors[:, t, :])))

            # add a time-axis
            score = expand_dims(e_t, axis=1)

            # apply softmax
            attention = softmax(score, axis=1)

            # cache all the attention weights at all timesteps
            attention_weights = concat([attention_weights, attention], axis=1)

        # calculate the CONTEXT vector
        context_vector = attention_weights * values
        # context_vector = reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights, coverage_vectors
