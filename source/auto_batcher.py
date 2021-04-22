import config
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences


class AutoBatcher:
    def __init__(self, input_path_code, input_path_ast, input_path_nl):
        self.paths = {
            "code": input_path_code,
            "ast": input_path_ast,
            "nl": input_path_nl
        }

        self.vocab_paths = {
            "code": config.CODE_VOCAB,
            "ast": config.AST_VOCAB,
            "nl": config.NL_VOCAB
        }

        self.vocab_sizes = dict()

    def get_vocab_dict(self, path):
        with open(path, 'r') as content_file:
            data = content_file.read().split('\n')
        return {k: v for (v, k) in enumerate(data)}

    def get_vocab_sizes(self):
        return self.vocab_sizes

    def preprocess(self, ext):
        vocab_dict = self.get_vocab_dict(self.vocab_paths[ext])
        self.vocab_sizes[ext] = len(vocab_dict)

        with open(self.paths[ext], 'r') as input_file:
            complete_dataset = []
            for line in input_file.readlines():
                line = f"<S> {line} </S>"
                indices = []
                for word in line.split():
                    try:
                        indices.append(vocab_dict[word])
                    except KeyError:
                        indices.append(config.UNK)
                complete_dataset.append(indices)
            return pad_sequences(complete_dataset)

    def create_batched_dataset(self, batch_size):
        ast_data = self.preprocess("ast")
        nl_data = self.preprocess("nl")
        buffer_size = nl_data.shape[0]
        code_data = pad_sequences(self.preprocess("code"), maxlen=ast_data.shape[1])
        print(code_data.shape, ast_data.shape, nl_data.shape)
        dataset = Dataset.from_tensor_slices((code_data, ast_data, nl_data))
        return dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)