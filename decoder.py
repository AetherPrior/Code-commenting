class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_len_inp):
    super(Decoder, self).__init__()
    
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    
    # cuDNN implementation compatible LSTM
    
    self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    x = self.embedding(x)
    
    context_vector, attention_weights, coverage_vector = self.attention(hidden, enc_output)
    
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    
    output, _, state = self.lstm(x)
    output = tf.reshape(output, (-1, output.shape[2]*output.shape[1]))
    
    x = tf.nn.softmax(self.fc(output))
    
    return x, state, attention_weights, coverage_vector
