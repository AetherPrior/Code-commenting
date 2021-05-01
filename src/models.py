import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow_addons.activations import mish
from tensorflow.nn import sigmoid, tanh, softmax
from tensorflow.keras.layers import LSTM, Dense, Embedding, Layer, Bidirectional


class BiEncoder(Model):
    def __init__(self, inp_dim, 
                 embed_dim,
                 enc_units):
        '''
        h_i - hidden vectors
        state_c - cell state output
        '''
        super(BiEncoder, self).__init__()

        self.embedding = Embedding(input_dim=inp_dim, 
                                   output_dim=embed_dim,
                                   mask_zero=True)
        self.lstm = LSTM(enc_units, 
                         return_state=True, 
                         return_sequences=True)
                         
        self.lstm = Bidirectional(self.lstm, 
                                  merge_mode="concat")
        
        self.reduce_c = Dense(enc_units*2)
        self.reduce_h = Dense(enc_units*2)

    def call(self, x):
        x = self.embedding(x)
        enc_output, forward_h, forward_c, backward_h, backward_c = self.lstm(x)
        state_c = mish(self.reduce_c(tf.concat([forward_c, backward_c], axis=-1)))
        state_h = mish(self.reduce_h(tf.concat([forward_h, backward_h], axis=-1)))
        return (enc_output, state_h, state_c)


class DeepCom(Model):
    def __init__(self, inp_dim_code, inp_dim_ast, embed_dim=128, enc_units=32):
        super(DeepCom, self).__init__()
        self.encoder_code = BiEncoder(inp_dim_code, embed_dim, enc_units)
        self.encoder_ast = BiEncoder(inp_dim_ast, embed_dim, enc_units)

    def call(self, x_c, x_a):
        hidden_c, state_h_c, state_c_c = self.encoder_code(x_c)
        hidden_a, state_h_a, state_c_a = self.encoder_ast(x_a)

        hidden = tf.concat([hidden_a, hidden_c], axis=-1)
        state_h = tf.concat([state_h_c, state_h_a], axis=-1)
        state_c = tf.concat([state_c_c, state_c_a], axis=-1)
        return (hidden, state_h, state_c)


class BahdanauAttention(Layer):
    def __init__(self, attn_sz):
        '''
        attn_sz = hidden[2], the decoder units
        '''
        super(BahdanauAttention, self).__init__()
        self.attn_sz = attn_sz

        self.V = Dense(1, use_bias=False)
        self.W_h = Dense(self.attn_sz, use_bias=False)
        self.W_c = Dense(self.attn_sz, use_bias=False)
        self.W_s = Dense(self.attn_sz)

    def call(self, h_i, s_t, coverage=None):
        '''
        h_i - encoder states: (batch_sz, t_k, attn_sz)
        s_t - decoder cell state: (batch_sz, state_sz)
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
    def __init__(self, inp_dim,
                 embed_dim=128, 
                 dec_units=128):
        '''
        attn_shape is same as enc_out_shape: h_i shape
        inp_dim == out_dim
        '''
        super(AttentionDecoder, self).__init__()
        self.attention = BahdanauAttention(attn_sz=dec_units)
        self.embedding = Embedding(inp_dim, embed_dim, mask_zero=True)
        self.lstm = LSTM(dec_units, 
                         return_state=True,
                         return_sequences=True) 

        self.W1 = Dense(1)
        self.W2 = Dense(dec_units)
        self.V1 = Dense(dec_units)
        self.V2 = Dense(inp_dim)
        self.prev_context_vector = None

    def call(self, x, h_i, prev_h, prev_c, coverage, max_oovs, code_ext):
        if self.prev_context_vector is None:
            context_vector, _, _ = self.attention(h_i, prev_h)
        else:
            context_vector = self.prev_context_vector
        
        x = self.embedding(x)
        x = self.W2(tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=1))
        _, state_h, state_c = self.lstm(x, initial_state=[prev_h, prev_c])
        
        context_vector, attn_dist, coverage = self.attention(h_i, state_h, coverage)
        p_vocab = softmax(self.V2(self.V1(tf.concat([context_vector, state_h], axis=1))))
        self.prev_context_vector = context_vector
        
        #----------------------------------**major-changes**-------------------------------#
        #-----------------------------------**pointer-gen**--------------------------------#
        
        batch_sz = x.shape[0]
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[-1]))
        p_gen = sigmoid(self.W1(tf.concat([context_vector, state_h, x], axis=-1)))
        
        P1 = p_gen * p_vocab
        P2 = (1.0 - p_gen) * attn_dist
        
        concat_extra_zeros = tf.zeros((batch_sz, max_oovs))
        P1 = tf.concat([P1, concat_extra_zeros], axis=-1)
        batch_nums = tf.expand_dims(tf.range(0, batch_sz), 1)
        batch_nums = tf.tile(batch_nums, [1, code_ext.shape[1]])
        indices = tf.stack((batch_nums, code_ext), axis=2)
        final_dist = P1 + tf.scatter_nd(indices, P2, [batch_sz, P1.shape[1]])

        return (final_dist, attn_dist, state_h, state_c, coverage)