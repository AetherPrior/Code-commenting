import sys
import config
import numpy as np
import tensorflow as tf
from loss import coverage_loss
from vocab2dict import VocabData
from tensorflow.keras.optimizers import Nadam, Adadelta, RMSprop
from models import BiEncoder, BahdanauAttention, AttentionDecoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_path_code = "./Dataset/data_RQ1/train/train.token.code"
input_path_nl = "./Dataset/data_RQ1/train/train.token.nl"
input_path_ast = "./Dataset/data_RQ1/train/train.token.ast"
vocab_size_code = 30000
vocab_size_nl = 30000
vocab_size_ast = 65


def preprocess(input_path, ext):
    if ext == "code":
        vocab = VocabData(config.CODE_VOCAB)
    elif ext == "nl":
        vocab = VocabData(config.NL_VOCAB)
    elif ext == "ast":
        vocab = VocabData(config.AST_VOCAB)

    with open(input_path, 'r') as input_file:
        for line in input_file.readlines():
            line = f"<S> {line} </S>"
            indices = []
            for word in line.split():
                try:
                    indices.append(vocab.vocab_dict[word])
                except KeyError:
                    indices.append(config.UNK)
            yield indices


def get_batch(batch_sz):
    batch_code, batch_nl, batch_ast = [], [], []

    gen_code, gen_nl, gen_ast = (preprocess(input_path_code, ext="code"),
                                 preprocess(input_path_nl, ext="nl"),
                                 preprocess(input_path_ast, ext="ast"))


    try:
        for i in range(batch_sz):
            batch_code.append(next(gen_code))
            batch_nl.append(next(gen_nl))
            batch_ast.append(next(gen_ast))
    except StopIteration:
        print("Max Size Reached")

    return (tf.convert_to_tensor(pad_sequences(batch_code)), 
            tf.convert_to_tensor(pad_sequences(batch_ast)),
            tf.convert_to_tensor(pad_sequences(batch_nl)))


def main():
    learning_rate = 1e-3
    batch_sz = 8

    encoder_code = BiEncoder(inp_dim=vocab_size_code)
    encoder_ast = BiEncoder(inp_dim=vocab_size_ast)

    decoder = AttentionDecoder(
        batch_sz=batch_sz,
        inp_dim=vocab_size_nl,
        out_dim=vocab_size_nl
    )

    optimizer = Adadelta(learning_rate=learning_rate)

    def train_step(inp_code, inp_ast, target):
        total_loss = 0
        with tf.GradientTape() as tape:
            hidden_state_code, cell_state_code = encoder_code(inp_code)
            hidden_state_ast, cell_state_ast = encoder_ast(inp_ast)
            
            hsct = hidden_state_code.shape[1]
            hsat = hidden_state_ast.shape[1]
            
            if hsct > hsat:
                hidden_state_ast = pad_sequences(hidden_state_ast, maxlen=hsct)
            else:
                hidden_state_code = pad_sequences(hidden_state_code, maxlen=hsat)

            hidden_state = tf.concat([hidden_state_code, hidden_state_ast], axis=-1)
            cell_state = tf.concat([cell_state_code, cell_state_ast], axis=-1)

            print(hidden_state.shape, cell_state.shape)
            
            dec_inp = tf.expand_dims([config.BOS] * batch_sz, axis=1)
            coverage = None

            for i in range(1, target.shape[1]):
                # teacher-forcing during training.
                # this means, pass the true output instead of the 
                # previous output to the decoder.
                cell_state, p_vocab, p_gen, attn_dist, coverage = decoder(
                    dec_inp, hidden_state, cell_state, coverage)

                p_vocab = p_gen*p_vocab
                p_attn = (1-p_gen)*attn_dist
                dec_inp = tf.expand_dims(target[:, i], 1)
                loss_value = coverage_loss(target[:, i], p_vocab, attn_dist, coverage)
                total_loss += loss_value
            batch_loss = total_loss / int(target.shape[1])
            trainable_var = encoder_ast.trainable_variables + \
                            encoder_code.trainable_variables + \
                            decoder.trainable_variables
            print("computing gradients")
            grads = tape.gradient(total_loss, trainable_var)
            print("applying gradients")
            optimizer.apply_gradients(zip(grads, trainable_var))
            return batch_loss


    for _ in range(3):
        inp_code, inp_ast, target = get_batch(batch_sz)
        print(train_step(inp_code, inp_ast, target).numpy())


if __name__ == '__main__':
    main()