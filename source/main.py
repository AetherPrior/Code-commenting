from vocab2dict import VocabData
from models import BiEncoder, BahdanauAttention, AttentionDecoder
import config
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset
from tensorflow import Tensor, convert_to_tensor
import tensorflow as tf
import numpy as np
from loss import coverage_loss

input_path_code = "./Dataset/data_RQ1/train/train.token.code"
input_path_nl = "./Dataset/data_RQ1/train/train.token.nl"
vocab_size_code = 30000
vocab_size_nl = 30000


def preprocess_nl(input_path, vocab_size_nl):
    vocab = VocabData(config.NL_VOCAB)
    vocab_size_nl = len(vocab.vocab_dict)

    with open(input_path, 'r') as input_file:
        for nl_line in input_file.readlines():
            indices = []
            for word in nl_line.split():
                try:
                    indices.append(vocab.vocab_dict[word])
                except KeyError:
                    indices.append(config.UNK)
            yield indices


def preprocess_code(input_path, vocab_size_code):
    vocab = VocabData(config.CODE_VOCAB)
    vocab_size_code = len(vocab.vocab_dict)

    with open(input_path, 'r') as input_file:
        for code_line in input_file.readlines():
            indices = []
            for word in code_line.split():
                try:
                    indices.append(vocab.vocab_dict[word])
                except KeyError:
                    indices.append(config.UNK)
            yield indices


def get_batch(batch_sz):
    batch_code, batch_nl = [], []

    gen_code, gen_nl = (preprocess_code(input_path_code, vocab_size_code),
                        preprocess_nl(input_path_nl, vocab_size_nl))

    try:
        for i in range(batch_sz):
            batch_code.append(next(gen_code))
            batch_nl.append(next(gen_nl))

    except StopIteration:
        print("Max Size Reached")

    # gen2 = preprocess_nl(input_path_nl, vocab_size_nl)
    return convert_to_tensor(pad_sequences(batch_code)), convert_to_tensor(pad_sequences(batch_nl))


def main():
    batch_sz = 8

    # print(batch.shape)

    encoder = BiEncoder(inp_dim=vocab_size_code+1)

    # print(hidden_state.shape)

    decoder = AttentionDecoder(
        batch_sz=batch_sz,
        inp_dim=(vocab_size_nl+1),
        out_dim=(vocab_size_nl+1)
    )

    @tf.function
    def train_step(batch, target):
        hidden_state, cell_state = encoder(batch)
        coverage = None

        for i in range(1, target.shape[1]):
            # teacher-forcing
            cell_state, p_vocab, p_gen, attn_dist, coverage = decoder(
                tf.expand_dims(target[i], axis=1), hidden_state, cell_state, coverage)

            p_vocab = p_gen*p_vocab
            p_oov = (1-p_gen)*attn_dist

    for i in range(3):
        batch, target = get_batch(batch_sz)
        train_step(batch, target)
    #x = convert_to_tensor(np.zeros((batch_sz, vocab_size_nl+1)))
    #decoder(x, h_i=hidden_state, state_c=cell_state)


if __name__ == '__main__':
    main()
