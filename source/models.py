import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.nn import sigmoid, tanh, softmax
from tensorflow.keras.layers import LSTM, Dense, Embedding, Layer, Bidirectional, BatchNormalization


class Encoder(Model):
    def __init__(self, inp_dim,
                 embed_dim=1024,
                 enc_units=256):
        '''
        h_i - hidden vectors
        state_c - cell state output
        '''
        super(Encoder, self).__init__()

        self.embedding = Embedding(input_dim=inp_dim,
                                   output_dim=embed_dim)
        self.lstm = LSTM(enc_units,
                         return_state=True,
                         return_sequences=True,
                         stateful=True)

        self.lstm = Bidirectional(self.lstm, merge_mode="concat")

    def call(self, x):
        x = self.embedding(x)
        enc_output, _, forward_c, _, backward_c = self.lstm(x)
        state_c = tf.concat([forward_c, backward_c], axis=-1)
        return (enc_output, state_c)


class BahdanauAttention(Layer):
    def __init__(self, batch_sz, attn_sz):
        '''
        batch_sz = enc_out_shape[0]
        attn_sz = enc_out_shape[2], the encoder units
        '''
        super(BahdanauAttention, self).__init__()
        self.batch_sz = batch_sz
        self.attn_sz = attn_sz

        self.V = Dense(1, use_bias=False)
        self.W_h = Dense(self.attn_sz, use_bias=False)
        self.W_c = Dense(self.attn_sz, use_bias=False)
        self.W_s = Dense(self.attn_sz)

    def call(self, h_i, s_t, coverage=None):
        '''
        h_i - encoder states: (batch_sz, t_k, attn_sz)
        s_t - decoder cell state: (batch_sz, cell_state_sz)
        '''
        enc_features = self.W_h(h_i)
        dec_features = tf.expand_dims(self.W_s(s_t), axis=1)
        features = enc_features + dec_features

        if coverage is not None:
            coverage_input = tf.expand_dims(coverage, axis=-1)
            features += self.W_c(coverage_input)

        e_t = self.V(tanh(features))
        e_t = tf.reshape(e_t, [-1, e_t.shape[1]])
        a_t = softmax(e_t, axis=1)

        if coverage is None:
            coverage = a_t
        else:
            coverage += a_t

        context_vector = tf.matmul(tf.expand_dims(a_t, axis=1), h_i)
        context_vector = tf.squeeze(context_vector, axis=1)
        return (context_vector, a_t, coverage)


class AttentionDecoder(Model):
    def __init__(self, batch_sz, inp_dim, out_dim,
                 embed_dim=1024,
                 dec_units=1024):
        '''
        attn_shape is same as enc_out_shape: h_i shape
        '''
        super(AttentionDecoder, self).__init__()
        self.attention = BahdanauAttention(
            batch_sz=batch_sz, attn_sz=dec_units)
        self.embedding = Embedding(inp_dim, embed_dim)
        self.lstm = LSTM(dec_units,
                         return_state=True,
                         stateful=True)

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

        _, state_h, state_c = self.lstm(x)

        # call Bahdanau's attention
        context_vector, attn_dist, coverage = self.attention(
            h_i, state_c, prev_coverage)

        # pass through 2 linear layers
        z = tf.concat([context_vector, state_h], axis=1)
        p_vocab = softmax(self.V2(self.V1(z)))

        # concat x and context vector
        x = tf.reshape(x, (-1, x.shape[1]*x.shape[-1]))
        temp = tf.concat([context_vector, state_c, x], axis=-1)

        # embed the concatenated vectors as the p_gen
        p_gen = sigmoid(self.W1(temp))

        # cache context vector
        self.prev_context_vector = context_vector

        return (state_c, p_vocab, p_gen, attn_dist, coverage)
