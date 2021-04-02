from vocab2dict import VocabData
from encoder import Encoder
from config import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset
from tensorflow import convert_to_tensor

input_path = "./Dataset/data_RQ1/train/train.token.code"
vocab_size = None


def preprocess(input_path):
    global vocab_size
    vocab = VocabData(CODE_VOCAB)
    vocab_size = len(vocab.vocab_dict)

    with open(input_path, 'r') as input_file:
        for code_line in input_file.readlines():
            indices = []
            for word in code_line.split():
                try:
                    indices.append(vocab.vocab_dict[word])
                except KeyError:
                    indices.append(2)
            yield indices


def get_batch(batch_sz):
    batch = []
    gen = preprocess(input_path)
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
    print(output, state)


if __name__ == '__main__':
    main()
