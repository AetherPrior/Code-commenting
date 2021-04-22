import config
import argparse
from time import time
import tensorflow as tf
from loss import coverage_loss
from vocab2dict import VocabData
from tensorflow.data import Dataset
from models import DeepCom, AttentionDecoder
from tensorflow_addons.optimizers import Lookahead
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
parser.add_argument("-log", "--logging", type=int, dest="logging",
                    default=100, help="log the loss after x batches")
parser.add_argument("-s", "--save", type=int, default=100,
                    dest="save", help="Save the model after x epochs")
parser.add_argument("-sd", "--savedir", type=str,
                    default="./saved_ckpts", help="Save directory")
parser.add_argument("-cov", "--coverage", dest="cov", type=float,
                    default=0.5, help="coverage loss hyperparameter")
parser.add_argument("-la", "--look-ahead", dest="la", action="store_true")
args = parser.parse_args()

avail_optims = {
    "sgd": SGD(learning_rate=args.lr, momentum=0.9, nesterov=True),
    "rmsprop": RMSprop(learning_rate=args.lr),
    "adam": Adam(learning_rate=args.lr, amsgrad=True),
    "nadam": Nadam(learning_rate=args.lr),
}


input_path_code = "../Dataset/data_RQ1/train/train.token.code"
input_path_nl = "../Dataset/data_RQ1/train/train.token.nl"
input_path_ast = "../Dataset/data_RQ1/train/train.token.ast"

vocab_size_code = 30000
vocab_size_nl = 30000
vocab_size_ast = 65


def sentence_oov(input_path, ext):
    '''
    For the pointer generator, it's better to use this function
    Yields a singular batch with its OOV tokens when called
    '''
    if ext == "code":
        vocab = VocabData(config.CODE_VOCAB)
    elif ext == "nl":
        vocab = VocabData(config.NL_VOCAB)
    elif ext == "ast":
        vocab = VocabData(config.AST_VOCAB)

    with open(input_path, 'r') as input_file:
        oov_tokens = []
        for line in input_file.readlines():
            line = f"<S> {line} </S>"
            indices = []
            for word in line.split():
                try:
                    indices.append(vocab.vocab_dict[word])
                except KeyError:
                    oov_tokens.append(word)
                    indices.append(config.UNK)
            yield indices, oov_tokens


def get_batch(batch_sz):
    '''
    Fetches an entire batch of code with OOV tokens
    '''
    batch_code, batch_nl, batch_ast = [], [], []

    gen_code, gen_nl, gen_ast = (sentence_oov(input_path_code, ext="code"),
                                 sentence_oov(input_path_nl, ext="nl"),
                                 sentence_oov(input_path_ast, ext="ast"))
    try:
        for i in range(batch_sz):
            code, oov_code = next(gen_code)
            ast, oov_ast = next(gen_ast)
            nl, oov_nl = next(gen_nl)

            batch_code.append(code)
            batch_ast.append(ast)
            batch_nl.append(nl)

    except StopIteration:
        print("[INFO] Max Size Reached")

    return (tf.convert_to_tensor(pad_sequences(batch_code)),
            tf.convert_to_tensor(pad_sequences(batch_ast)),
            tf.convert_to_tensor(pad_sequences(batch_nl))), (oov_code, oov_ast, oov_nl)


def preprocess(input_path, ext):
    if ext == "code":
        vocab = VocabData(config.CODE_VOCAB)
    elif ext == "ast":
        vocab = VocabData(config.AST_VOCAB)
    elif ext == "nl":
        vocab = VocabData(config.NL_VOCAB)

    with open(input_path, 'r') as input_file:
        oov_tokens = []
        complete_dataset = []
        for line in input_file.readlines():
            line = f"<S> {line} </S>"
            indices = []
            for word in line.split():
                try:
                    indices.append(vocab.vocab_dict[word])
                except KeyError:
                    oov_tokens.append(word)
                    indices.append(config.UNK)
            complete_dataset.append(indices)
        return pad_sequences(complete_dataset), oov_tokens


def create_batched_dataset(batch_size):
    ast_data, oov_ast = preprocess(input_path_ast, ext="ast")

    code_data, oov_code = preprocess(input_path_code, ext="code")
    code_data = pad_sequences(code_data, maxlen=ast_data.shape[1])

    nl_data, oov_nl = preprocess(input_path_nl, ext="nl")

    buffer_size = code_data.shape[0]
    dataset = Dataset.from_tensor_slices(
        (code_data, ast_data, nl_data)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return (dataset, buffer_size)


def main():
    batch_sz = args.bs
    logging = args.logging
    save = args.save
    savedir = args.savedir

    dataset, buffer_sz = create_batched_dataset(batch_sz)

    encoder = DeepCom(inp_dim_code=vocab_size_code,
                      inp_dim_ast=vocab_size_ast)

    decoder = AttentionDecoder(inp_dim=vocab_size_nl)

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
                p_vocab, p_gen, attn_dist, state_h, state_c, coverage = decoder(dec_inp, hidden,
                                                                                prev_h=state_h,
                                                                                prev_c=state_c)

                p_vocab = p_gen * p_vocab
                p_attn = (1-p_gen) * attn_dist
                total_loss += criterion(targ, p_vocab,
                                        attn_dist,
                                        coverage,
                                        args.cov)

            batch_loss = total_loss / target.shape[1]
            trainable_var = encoder.trainable_variables + decoder.trainable_variables

            grads = tape.gradient(total_loss, trainable_var)
            optim.apply_gradients(zip(grads, trainable_var))
        return tf.reduce_sum(batch_loss)

    print(f"[INFO] Steps per epoch: {buffer_sz // batch_sz}")
    avg_time_per_batch = 0

    #    ckpt = tf.train.Checkpoint(step=tf.Variable(initial_value=0),
    #                               optimizer=optim,
    #                               encoder=encoder,
    #                               decoder=decoder)
    #    manager = tf.train.CheckpointManager(
    #        ckpt, savedir, max_to_keep=10)
    #
    #    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    #    if manager.latest_checkpoint:
    #        print(f"[INFO] Restored from {manager.latest_checkpoint}")
    #    else:
    #        print("Initializing from scratch")

    for epoch in range(1, args.epochs+1):
        print(f"[INFO] Running epoch: {epoch}")
        for (batch, (code, ast, targ)) in enumerate(dataset):
            start = time()
            batch_loss = train_step(code, ast, targ).numpy()
            avg_time_per_batch += (time() - start)
            if epoch == 1 and not batch:
                print(encoder.summary())
                print(decoder.summary())
            if not batch % logging:
                print("[INFO] Batch: {} | Loss: {:.2f} | {:.2f} sec/batch".format(batch,
                                                                                  batch_loss, avg_time_per_batch/logging))
                avg_time_per_batch = 0
            # if not batch % save:
            #    ckpt.step.assign_add(100)
            #    save_path = manager.save()
            #    print(
            #        f"[LOGGING] Saved checkpoint for step: {int(ckpt.step)} at {save_path}")


if __name__ == '__main__':
    main()
