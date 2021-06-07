import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.nn import softmax, sigmoid, tanh
from tensorflow.keras.layers import LSTM, Dense, Embedding, Layer


class Encoder(Model):
    def __init__(self, inp_dim, 
                 embed_dim,
                 enc_units):
        '''
        h_i - hidden vectors
        state_c - cell state output
        state_h - hidden state output
        '''
        super(Encoder, self).__init__()
        self.embedding = Embedding(input_dim=inp_dim, 
                                   output_dim=embed_dim,
                                   mask_zero=True)
        
        self.lstm1 = LSTM(enc_units,
                          return_state=True, 
                          return_sequences=True)
                         
        self.lstm2 = LSTM(enc_units,
                          return_state=True, 
                          return_sequences=True)

    def call(self, x):
        x = self.embedding(x)
        x, _, _ = self.lstm1(x)
        x, state_h, state_c = self.lstm2(x)
        return (x, state_h, state_c)


class DeepComEncoder(Model):
    def __init__(self, inp_dim_code, 
                 inp_dim_ast, 
                 embed_dim=512, 
                 enc_units=128):
        '''
        contains both code 
        and AST encoder
        '''
        super(DeepComEncoder, self).__init__()
        self.encoder_code = Encoder(inp_dim_code, embed_dim, enc_units)
        self.encoder_ast = Encoder(inp_dim_ast, embed_dim, enc_units)
        self.W1 = Dense(enc_units)
        self.W2 = Dense(enc_units)
        self.W3 = Dense(enc_units)

    def call(self, x_c, x_a):
        '''
        x_c = batch x timestep
        x_a = batch x timestep
        '''
        hidden_c, state_h_c, state_c_c = self.encoder_code(x_c) 
        hidden_a, state_h_a, state_c_a = self.encoder_ast(x_a)

        hidden = self.W1(tf.concat([hidden_a, hidden_c], axis=-1))
        state_h = self.W2(tf.concat([state_h_c, state_h_a], axis=-1))
        state_c = self.W3(tf.concat([state_c_c, state_c_a], axis=-1))
        return (hidden, state_h, state_c)


class BahdanauAttention(Layer):
    def __init__(self, attn_sz):
        '''
        attn_sz = hidden[2], the decoder units
        '''
        super(BahdanauAttention, self).__init__()
        self.attn_sz = attn_sz

        self.V = Dense(1, use_bias=False)     
        self.W_c = Dense(self.attn_sz, use_bias=False)
        self.W_h = Dense(self.attn_sz, use_bias=False) 
        self.W_s = Dense(self.attn_sz)                 

    def call(self, h_i, s_t, coverage=None):
        '''
        h_i - encoder states - batch_sz x timesteps x 2*enc_units
        s_t - decoder state - batch_sz x 2*dec_units
        '''
        enc_features = self.W_h(h_i)                         
        dec_features = tf.expand_dims(self.W_s(s_t), axis=1) 
        features = enc_features + dec_features 
    
        if coverage is not None:
            features += self.W_c(tf.expand_dims(coverage, axis=-1))
            
        e_t = self.V(tanh(features))
        e_t = tf.squeeze(e_t, axis=-1)
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
                 embed_dim=512, 
                 dec_units=128):
        '''
        attn_shape is same as enc_out_shape: h_i shape
        inp_dim == out_dim == 30001
        '''
        super(AttentionDecoder, self).__init__()
        self.attention = BahdanauAttention(dec_units)
        self.embedding = Embedding(inp_dim, 
                                   embed_dim, 
                                   mask_zero=True)
        
        self.lstm = LSTM(dec_units,
                         return_state=True,
                         return_sequences=True)
        
        self.W1 = Dense(embed_dim)
        self.W2 = Dense(1)
        self.V1 = Dense(dec_units*3)
        self.V2 = Dense(inp_dim)

    def call(self, x, h_i, prev_state, context_vector, coverage, max_oovs, code_ext):
        '''
        x - batch_sz x 1 x 30001
        h_i - batch_sz x timesteps x enc_units
        prev_state = [state_h, state_c]
        coverage - batch_sz x timesteps
        max_oovs - len(batch.oovs)
        code_ext - batch_sz x timesteps
        '''
        if context_vector is None:
            s_t = tf.concat(prev_state, axis=-1)
            context_vector, _, _ = self.attention(h_i, s_t)
        
        x = tf.squeeze(self.embedding(x))                
        x1 = self.W1(tf.concat([x, context_vector], axis=-1))
        
        cell_out, state_h, state_c = self.lstm(tf.expand_dims(x1, axis=1), 
                                               initial_state=prev_state)
                                               
        s_t = tf.concat([state_h, state_c], axis=-1)
        cell_out = tf.squeeze(cell_out, axis=1)
        
        context_vector, attn_dist, coverage = self.attention(h_i, s_t, coverage)
        p_vocab = softmax(self.V2(self.V1(tf.concat([cell_out, context_vector], axis=-1))))
    
        #-----------------------------------**pointer-gen**--------------------------------#
        
        batch_sz = x.shape[0]
        p_gen = sigmoid(self.W2(tf.concat([context_vector, s_t, x], axis=-1)))
        
        P1 = p_gen * p_vocab 
        P2 = (1.0 - p_gen) * attn_dist 
        
        concat_extra_zeros = tf.zeros((batch_sz, max_oovs))
        P1 = tf.concat([P1, concat_extra_zeros], axis=-1)
        
        batch_nums = tf.expand_dims(tf.range(0, batch_sz), axis=1)
        batch_nums = tf.tile(batch_nums, [1, code_ext.shape[1]])
        indices = tf.stack((batch_nums, code_ext), axis=2) 
        
        final_dist = P1 + tf.scatter_nd(indices, P2, [batch_sz, P1.shape[1]])
        return (final_dist, attn_dist, [state_h, state_c], coverage, context_vector)