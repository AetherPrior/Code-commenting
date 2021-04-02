from tensorflow.keras import Model
from tensorflow import concat, expand_dims, reshape
from tensorflow.keras.layers import Dense, Embedding, LSTM
from Bahdanau_Attention import BahdanauAttention
from tensorflow.nn import softmax


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

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        context_vector, attention_weights, coverage_vector = self.attention(hidden, enc_output)
        x = concat([expand_dims(context_vector, 1), x], axis=-1)
        output, _, state = self.lstm(x)
        output = reshape(output, (-1, output.shape[2]*output.shape[1]))
        x = softmax(self.fc(output))
        return x, state, attention_weights, coverage_vector
