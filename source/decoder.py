from tensorflow.keras import Model
from tensorflow import concat, expand_dims, reshape
from tensorflow.keras.layers import Dense, Embedding, LSTM
from Bahdanau_Attention import BahdanauAttention
from tensorflow.nn import softmax, sigmoid


class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_len_inp):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)

        # cuDNN implementation compatible LSTM

        self.lstm = LSTM(self.dec_units,
                         return_sequences=True,
                         return_state=True)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
        self.V1 = Dense(self.dec_units//2)
        self.V2 = Dense(vocab_size)

        self.W1 = Dense(1)
        self.W2 = Dense(1)
        self.W3 = Dense(1)

    def call(self, x, h_i):
        '''
        x = input
        h_i = encoder hidden states after ith input
        '''

        x = self.embedding(x)
        output, s_t, state = self.lstm(x)
        context_vector, attention_weights, coverage_vector = self.attention(
            s_t, h_i)

        # x = concat([expand_dims(context_vector, 1), x], axis=-1)

        # pointer
        p_vocab = softmax(
            self.V2(self.V1(concat([s_t, context_vector]), axis=-1)))

        # pointer
        p_gen = sigmoid(self.W1(context_vector) + self.W2(s_t) + self.W3(x))

        output = reshape(output, (-1, output.shape[2]*output.shape[1]))
        prediction = softmax(self.fc(output))
        return prediction, attention_weights, coverage_vector
