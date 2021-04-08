import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.nn import sigmoid, tanh, softmax
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, Conv2D


class BiEncoder(Model):
    def __init__(self, enc_inp_shape, inp_dim, embed_dim=1024, enc_units=1024):
        '''
        Perferable to have a BiLSTM
        h_i - hidden vectors
        state_c - cell state output
        '''
        super(BiEncoder, self).__init__()

        self.embedding = Embedding()
        self.lstm = LSTM(enc_units, return_state=True, return_sequences=True)
        self.bilstm = Bidirectional(self.lstm, merge_mode="concat")

    def call(self, x):
        x = self.embedding(x)
        h_i, _, state_c = self.bilstm(x)
        return (h_i, state_c)


class BahdanauAttention(Layer):
    def __init__(self, enc_out_shape):
        super(BahdanauAttention, self).__init__()
        self.batch_sz = enc_out_shape[0]
        self.attn_sz = enc_out_shape[2]

        self.W_h = Conv2D(filters=self.attn_sz, kernel_size=1, padding="same")
        self.W_c = Conv2D(filters=self.attn_sz, kernel_size=1, padding="same")
        self.W_s = Dense(self.attn_sz)
        self.V = Dense(self.attn_sz, use_bias=False)

    def call(self, h_i, s_t, coverage=None):
        '''
        h_i - encoder states: (batch_sz, attn_length, attn_size)
        s_t - decoder cell state: (batch_sz, cell_state_sz)
        '''
        h_i = tf.expand_dims(h_i, axis=2)

        enc_features = self.W_h(h_i)

        dec_features = self.W_s(s_t)
        dec_features = tf.expand_dims(tf.expand_dims(dec_features, 1), 1)

        if coverage is None:
            a_t = softmax(tf.reduce_sum(
                self.v(tanh(enc_features + dec_features)), axis=[2, 3]))
            coverage = tf.expand_dims(tf.expand_dims(a_t, 2), 2)

        else:
            cov_features = self.W_c(coverage)
            a_t = softmax(tf.reduce_sum(
                self.V(tanh(enc_features + dec_features + cov_features)), axis=[2, 3]))
            coverage += tf.reshape(a_t, [self.batch_sz, -1, 1, 1])

        context_vector = tf.reduce_sum(tf.reshape(
            a_t, [self.batch_sz, -1, 1, 1]) * h_i, axis=[1, 2])

        context_vector = tf.reshape(context_vector, [-1, self.attn_sz])

        return (context_vector, a_t, coverage)


class AttentionDecoder(Model):
    def __init__(self, attn_shape, inp_dim, out_dim, embed_dim=1024, dec_units=1024):
        '''
        attn_shape is same as enc_out_shape
        '''
        super(AttentionDecoder, self).__init__()
        self.attention = BahdanauAttention(attn_shape)
        self.embedding = Embedding(inp_dim, embed_dim)
        self.lstm = LSTM(dec_units, return_state=True)

        self.W1 = Dense(1)
        self.W2 = Dense(dec_units)
        self.V1 = Dense(dec_units)
        self.V2 = Dense(out_dim)

        self.prev_context_vector = None

    def call(self, x, h_i, state_c, prev_coverage=None):

        if self.prev_context_vector is None:
            context_vector, _, _ = self.attention(h_i, state_c)
        else:
            context_vector = self.prev_context_vector

        x = self.embedding(x)

        x = self.W2(
            tf.concat([x, tf.expand_dims(context_vector, axis=1)], axis=1))

        _, _, state_c = self.lstm(x)

        # call Bahdanau's attention
        context_vector, attn_dist, coverage = self.attention(
            h_i, state_c, prev_coverage)

        # pass through 2 linear layers
        p_vocab = softmax(
            self.V2(self.V1(tf.concat([context_vector, state_c], axis=1))))

        # concat x and context vector
        temp = tf.concat([context_vector, state_c, tf.reshape(
            x, (-1, x.shape[1]*x.shape[2]))], axis=-1)

        # embed the concatenated vectors
        p_gen = sigmoid(self.W1(temp))

        # reset context vector
        self.prev_context_vector = context_vector

        return (state_c, p_vocab, p_gen, attn_dist, coverage)
