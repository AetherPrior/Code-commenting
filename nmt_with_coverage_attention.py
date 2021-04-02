# -*- coding: utf-8 -*-
"""
nmt_with_coverage_attention
Borrowed from OpenNMT 
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from unicodedata import normalize, category
from re import sub
from os.path import dirname
from time import time
from encoder import Encoder
from decoder import Decoder
from Bahdanau_Attention import Bahdanau_Attention

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)


num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
    path_to_file, num_examples)
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2)

LOCAL_BATCH_SIZE = 8
embedding_dim = 256
units = 1024
BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train) // LOCAL_BATCH_SIZE
print(steps_per_epoch)
BATCH_SIZE = LOCAL_BATCH_SIZE * strategy.num_replicas_in_sync
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1
dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

with strategy.scope():
    def coverage_loss(targets, predictions, attention_weights, coverage_vector, coverage_lambda=0.5):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)
        covloss = tf.reduce_sum(tf.minimum(coverage_vector, attention_weights))
        return loss_object(targets, predictions) + coverage_lambda*(covloss)

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units,
                      BATCH_SIZE, max_len_inp=max_length_inp)
    loss_function = coverage_loss
    optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)


@tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, attention_weights, coverage_vector = decoder(
                dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions,
                                  attention_weights, coverage_vector)
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        total_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, total_variables)
        optimizer.apply_gradients(zip(gradients, total_variables))
    return batch_loss


EPOCHS = 25
LOSSES = []

for epoch in range(EPOCHS):
    start = time()
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss_as_tensor = strategy.run(train_step, args=(inp, targ))
        batch_loss = sum([a.numpy()
                          for a in batch_loss_as_tensor.values]) * (1./BATCH_SIZE)
        total_loss += batch_loss
        if not batch % 100:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss}')

    LOSSES.append(total_loss / steps_per_epoch)
    print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch}')
    print(f'Time taken for 1 epoch {time() - start} sec\n')

plt.plot(LOSSES, range(1, EPOCHS+1))
plt.xlabel("epochs")
plt.ylabel("losses")
plt.title("loss vs epochs")
plt.show()


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += f"{targ_lang.index_word[predicted_id]} "
        if targ_lang.index_word[predicted_id] == '<end>':
            break
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(sentence):
    result, sentence = evaluate(sentence)
    print(f'Input: {sentence}')
    print(f'Predicted translation: {result}'.format())
