from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM


class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.enc_units, return_sequences=True,
                         return_state=True, stateful=False)

    def call(self, x):
        x = self.embedding(x)
        # output, state_h, state_c
        output, _, state = self.lstm(x)
        return output, state
