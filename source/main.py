import config
import argparse
from time import time
import tensorflow as tf
from loss import coverage_loss
from vocab2dict import VocabData
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import *
from models import Encoder, AttentionDecoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_addons.optimizers import NovoGrad, SGDW, AdamW, Lookahead


# Some global declarations
parser = argparse.ArgumentParser(description="Run the Model")
parser.add_argument("-b", "--batch-size", dest="bs", type=int,
                    default=8, help="Batch size for the model")
parser.add_argument("-e", "--epochs", type=int, default=10,
                    help="Number of epochs for the model")
parser.add_argument("-lr", "--learning-rate", dest="lr",
                    type=float, default=1e-3, help="learning rate for the model")
parser.add_argument("-o", "--optimizer", type=str,
                    default="adadelta", help="type of optimizer")
parser.add_argument("-log", "--logging", type=int,
                    default=100, help="log the loss after")
parser.add_argument("-cov", "--coverage", dest="cov", type=float,
                    default=0.5, help="coverage loss hyperparameter")
parser.add_argument("-la", "--look-ahead", dest="la", action="store_true")
parser.add_argument("-wd", "--weight-decay", dest="wd", type=float,
                    default=1e-2, help="weight decay for SGDW, AdamW and NovoGrad")
args = parser.parse_args()

avail_optims = {
    "sgd": SGD(learning_rate=args.lr, momentum=0.9, nesterov=True),
    "adagrad": Adagrad(learning_rate=args.lr),
    "adadelta": Adadelta(learning_rate=args.lr),
    "rmsprop": RMSprop(learning_rate=args.lr),
    << << << < Updated upstream
    "adam": Adam(learning_rate=args.lr),
    == == == =
    "adam": Adam(learning_rate=args.lr, amsgrad=True),
    >>>>>> > Stashed changes
    "nadam": Nadam(learning_rate=args.lr),
    "novograd": NovoGrad(learning_rate=args.lr, weight_decay=args.wd),
    "adamw": AdamW(learning_rate=args.lr, weight_decay=args.wd, amsgrad=True),
    "sgdw": SGDW(learning_rate=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
}

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
        complete_dataset = []
        for line in input_file.readlines():
            line = f"<S> {line} </S>"
            indices = []
            for word in line.split():
                try:
                    indices.append(vocab.vocab_dict[word])
                except KeyError:
                    indices.append(config.UNK)
            complete_dataset.append(indices)
        return pad_sequences(complete_dataset)


def create_batched_dataset(batch_size):
    ast_data = preprocess(input_path_ast, ext="ast")
    code_data = preprocess(input_path_code, ext="code")
    code_data = pad_sequences(code_data, maxlen=ast_data.shape[1])
    nl_data = preprocess(input_path_nl, ext="nl")
    buffer_size = code_data.shape[0]
    print(code_data.shape, ast_data.shape, nl_data.shape)
    dataset = Dataset.from_tensor_slices(
        (code_data, ast_data, nl_data)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return (dataset, buffer_size)


def main():
    batch_sz = args.bs
    epochs = args.epochs
    logging = args.logging
    cov_lambda = args.cov
    dataset, buffer_sz = create_batched_dataset(batch_sz)

    encoder_code = Encoder(inp_dim=vocab_size_code)
    encoder_ast = Encoder(inp_dim=vocab_size_ast)

    decoder = AttentionDecoder(
        batch_sz=batch_sz,
        inp_dim=vocab_size_nl,
        out_dim=vocab_size_nl
    )

    print(f"[INFO] Using the optimizer {args.optimizer}")
    optim = avail_optims[args.optimizer]
    if args.la:
        print("[INFO] Using LookAhead wrapper on optimizer")
        optim = Lookahead(optim)
    criterion = coverage_loss

    def train_step(inp_code, inp_ast, target):
        total_loss = 0
        with tf.GradientTape() as tape:
            hidden_state_code, cell_state_code = encoder_code(inp_code)
            hidden_state_ast, cell_state_ast = encoder_ast(inp_ast)

            # hidden_state_code = pad_sequences(hidden_state_code, maxlen=hidden_state_ast.shape[1])
            hidden_state = tf.concat(
                [hidden_state_code, hidden_state_ast], axis=-1)
            cell_state = tf.concat([cell_state_code, cell_state_ast], axis=-1)
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
                loss_value = criterion(
                    targ, p_vocab, attn_dist, coverage, coverage_lambda=cov_lambda)
                total_loss += loss_value

            batch_loss = total_loss / target.shape[1]
            trainable_var = encoder_code.trainable_variables + \
                encoder_ast.trainable_variables + \
                decoder.trainable_variables

            grads = tape.gradient(total_loss, trainable_var)
            optim.apply_gradients(zip(grads, trainable_var))
        return tf.reduce_sum(batch_loss)

    print(f"[INFO] Steps per epoch: {buffer_sz // batch_sz}")
    avg_time_per_batch = 0

    for _ in range(1, epochs+1):
        print(f"[INFO] Running epoch: {_}")
        for (batch, (code, ast, targ)) in enumerate(dataset):
            start = time()
            batch_loss = train_step(code, ast, targ).numpy()
            avg_time_per_batch += (time() - start)
            if not batch % logging:
                print("[INFO] Batch: {} | Loss: {:.2f} | {:.2f} sec/batch".format(batch,
                                                                                  batch_loss, avg_time_per_batch/logging))
                avg_time_per_batch = 0


if __name__ == '__main__':
    main()
