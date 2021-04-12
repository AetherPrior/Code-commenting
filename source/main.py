import config
from time import time
import tensorflow as tf
import tensorboard
from datetime import datetime
from loss import coverage_loss
from vocab2dict import VocabData
from tensorflow.keras.optimizers import Adam
from models import BiEncoder, AttentionDecoder
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
    learning_rate = 5e-3
    batch_sz = 8

    encoder_code = BiEncoder(inp_dim=vocab_size_code)
    encoder_ast = BiEncoder(inp_dim=vocab_size_ast)

    decoder = AttentionDecoder(
        batch_sz=batch_sz,
        inp_dim=vocab_size_nl,
        out_dim=vocab_size_nl
    )

    optimizer = Adam(learning_rate=learning_rate, amsgrad=True)
    criterion = coverage_loss

    def train_step(inp_code, inp_ast, target):
        total_loss = 0
        with tf.GradientTape() as tape:
            hidden_state_code, cell_state_code = encoder_code(inp_code)
            hidden_state_ast, cell_state_ast = encoder_ast(inp_ast)

            hidden_state = tf.concat(
                [hidden_state_code, hidden_state_ast], axis=1)
            cell_state = cell_state_code + cell_state_ast
            coverage = None

            for i in range(1, target.shape[1]):
                dec_inp = tf.expand_dims(
                    (([config.BOS] * batch_sz) if (i == 1) else targ), axis=1)
                cell_state, p_vocab, p_gen, attn_dist, coverage = decoder(dec_inp,
                                                                          hidden_state,
                                                                          cell_state,
                                                                          coverage)

                targ = target[:, i]
                p_vocab = p_gen * p_vocab
                p_attn = (1-p_gen) * attn_dist
                loss_value = criterion(targ, p_vocab, attn_dist, coverage)
                total_loss += loss_value

            batch_loss = total_loss / int(target.shape[1])
            trainable_var = encoder_code.trainable_variables + \
                encoder_ast.trainable_variables + \
                decoder.trainable_variables

            grads = tape.gradient(total_loss, trainable_var)
            optimizer.apply_gradients(zip(grads, trainable_var))
        return tf.reduce_sum(batch_loss)

    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    for _ in range(10):
        start = time()
        inp_code, inp_ast, target = get_batch(batch_sz)
        batch_loss = train_step(inp_code, inp_ast, target)
        runtime = round(time() - start, 2)
        print(f"Loss: {batch_loss} Time: {runtime}s")


if __name__ == '__main__':
    main()
