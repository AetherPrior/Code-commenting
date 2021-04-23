import config
import argparse
from time import time
import tensorflow as tf
from loss import coverage_loss
from tensorflow.data import Dataset
from models import DeepCom, AttentionDecoder
from tensorflow_addons.optimizers import Lookahead
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD


# Some global declarations
parser = argparse.ArgumentParser(description="Run the Model")
parser.add_argument("-b", "--batch-size", dest="bs", type=int, default=8, help="Batch size for the model")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for the model")
parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, default=1e-3, help="learning rate for the model")
parser.add_argument("-o", "--optimizer", type=str, default="adagrad", help="type of optimizer")
parser.add_argument("-log", "--logging", type=int, dest="logging", default=100, help="log the loss after x batches")
parser.add_argument("-cov", "--coverage", dest="cov", type=float, default=0.5, help="coverage loss hyperparameter")
parser.add_argument("-la", "--look-ahead", dest="la", action="store_true")
args = parser.parse_args()


avail_optims = {
    "sgd": SGD(learning_rate=args.lr, momentum=0.9, nesterov=True),
    "rmsprop": RMSprop(learning_rate=args.lr),
    "adam": Adam(learning_rate=args.lr, amsgrad=True),
    "adagrad": Adagrad(learning_rate=args.lr)
}


def read_vocab(path):
    with open(path, 'r') as content_file:
        data = content_file.read().split('\n')
    return {k: v for (v, k) in enumerate(data)}


def preprocess(input_path, ext):
    if ext == "code":
        vocab = read_vocab(config.CODE_VOCAB)
    elif ext == "ast":
        vocab = read_vocab(config.AST_VOCAB)
    elif ext == "nl":
        vocab = read_vocab(config.NL_VOCAB)

    with open(input_path, 'r') as input_file:
        complete_dataset = []
        for line in input_file.readlines():
            indices = []
            line = f"<S> {line} </S>"
            for word in line.split():
                try:
                    indices.append(vocab[word])
                except KeyError:
                    indices.append(config.UNK)
            complete_dataset.append(indices)
        return pad_sequences(complete_dataset)


def create_batched_dataset(batch_size):
    ast = preprocess(config.AST_INPUT, ext="ast")
    (buffer_sz, maxlen) = ast.shape
    
    code = preprocess(config.CODE_INPUT, ext="code")
    code = pad_sequences(code, maxlen=maxlen)

    nl = preprocess(config.NL_INPUT, ext="nl")

    dataset = Dataset.from_tensor_slices((code, ast, nl))
    dataset = dataset.shuffle(buffer_sz)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return (dataset, buffer_sz)


def main():
    batch_sz = args.bs
    logging = args.logging
    
    dataset, buffer_sz = create_batched_dataset(batch_sz)

    encoder = DeepCom(inp_dim_code=config.vocab_size_code,
                      inp_dim_ast=config.vocab_size_ast)

    decoder = AttentionDecoder(inp_dim=config.vocab_size_nl)

    print(f"[INFO] Using the optimizer {args.optimizer}")
    optim = avail_optims[args.optimizer]
    criterion = coverage_loss
    
    if args.la:
        print("[INFO] Using LookAhead wrapper on optimizer")
        optim = Lookahead(optim)

    def train_step(inp_code, inp_ast, target):
        total_loss = 0
        with tf.GradientTape() as tape:
            hidden, state_h, state_c = encoder(inp_code, inp_ast)

            for i in range(1, target.shape[1]):
                targ = target[:, i]
                dec_inp = tf.expand_dims(
                    (([config.BOS] * batch_sz) if (i == 1) else targ), axis=1)
                p_vocab, attn_dist, state_h, state_c, coverage = decoder(dec_inp, 
                                                                         hidden,
                                                                         prev_h=state_h,
                                                                         prev_c=state_c)

                # p_vocab = p_gen * p_vocab
                # p_attn = (1-p_gen) * attn_dist
                total_loss += criterion(targ, p_vocab,
                                        attn_dist,
                                        coverage,
                                        args.cov)

            batch_loss = total_loss / target.shape[1]
            variables = encoder.trainable_variables + \
                        decoder.trainable_variables

            grads = tape.gradient(total_loss, variables)
            optim.apply_gradients(zip(grads, variables))
        return tf.reduce_sum(batch_loss)

    print(f"[INFO] Steps per epoch: {buffer_sz // batch_sz}")
    time_per_batch = 0

    for epoch in range(1, args.epochs+1):
        print(f"[INFO] Running epoch: {epoch}")
        for (n_batch, (code, ast, targ)) in enumerate(dataset):
            start = time()
            batch_loss = train_step(code, ast, targ)
            time_per_batch += (time() - start)
            if epoch == 1 and not n_batch:
                print(encoder.summary())
                print(decoder.summary())
            if not n_batch % logging:
                print("[INFO] Batch: {} | Loss: {:.2f} | {:.2f} sec/batch".format(n_batch,
                                                                                  batch_loss.numpy(), 
                                                                                  time_per_batch/logging))
                time_per_batch = 0


if __name__ == '__main__':
    main()
