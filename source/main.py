import sys
import config
import numpy as np
import tensorflow as tf
# from loss import coverage_loss
from vocab2dict import VocabData
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
    batch_sz = 8

    encoder_code = BiEncoder(inp_dim=vocab_size_code)
    encoder_ast = BiEncoder(inp_dim=vocab_size_ast)

    decoder = AttentionDecoder(
        batch_sz=batch_sz,
        inp_dim=(vocab_size_nl),
        out_dim=(vocab_size_nl)
    )

    def train_step(inp_code, inp_ast, target):
        hidden_state_code, cell_state_code = encoder_code(inp_code)
        hidden_state_ast, cell_state_ast = encoder_ast(inp_ast)

        print(hidden_state_code.shape, cell_state_code.shape)
        print(hidden_state_ast.shape, cell_state_ast.shape)
        sys.exit(0)

        dec_inp = tf.expand_dims(
            [config.BOS] * batch_sz, 1)
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

    for _ in range(3):
        inp_code, inp_ast, target = get_batch(batch_sz)
        train_step(inp_code, inp_ast, target)


if __name__ == '__main__':
    main()