from vocab2dict import VocabData
from encoder import Encoder
import config
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset
from tensorflow import convert_to_tensor

input_path_code = "./Dataset/data_RQ1/train/train.token.code"
input_path_nl = "./Dataset/data_RQ1/train/train.token.nl"
vocab_size_code = vocab_size_nl = None


def preprocess_nl(input_path, vocab_size_nl):
    vocab = VocabData(config.CODE_VOCAB)
    vocab_size_code = len(vocab.vocab_dict)

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
    batch = []
    gen = preprocess_code(input_path_code, vocab_size_code)
    try:
        for i in range(batch_sz):
            batch.append(next(gen))
    except StopIteration:
        print("Max Size Reached")
    return batch


def main():
    batch_sz = 128
    batch = convert_to_tensor(pad_sequences(get_batch(batch_sz)))
    encoder = Encoder(vocab_size=vocab_size,
                      batch_sz=batch_sz,
                      embedding_dim=1024,
                      enc_units=1024)

    output, state = encoder(batch)
    decoder = Decoder(

    )


if __name__ == '__main__':
    main()
