import config
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Batch:
    def __init__(self):
        self.nl = None
        self.ast = None
        self.code = None
        self.nl_ex = None
        self.code_ex = None
        self.code_oovs = None
<<<<<<< Updated upstream:src-tf/DataWorks.py
        self.max_code_oovs = None
=======
>>>>>>> Stashed changes:src/utils.py


class BatchQueue:
    def __init__(self, batch_sz, key="train"):
        self.key = key
        self.batch_sz = batch_sz
        self.__ast_vocab = self.__read_vocab(config.AST_VOCAB)
        self.__code_vocab = self.__read_vocab(config.CODE_VOCAB)
        self.__nl_vocab, self.nl_list = self.__read_vocab(config.NL_VOCAB, return_data=True)

        self.__nl_data = self.__read_data(config.paths[self.key]["NL_INPUT"])
        self.__ast_data = self.__read_data(config.paths[self.key]["AST_INPUT"])
        self.__code_data = self.__read_data(config.paths[self.key]["CODE_INPUT"])
        
<<<<<<< Updated upstream:src-tf/DataWorks.py
        self.__fill_UNK = lambda lst, vocab: [vocab["<UNK>"] if w > len(vocab) else w for w in lst]
=======
        self.__fill_UNK = lambda lst, vocab: [vocab["<UNK>"] if w >= len(vocab) else w for w in lst]
>>>>>>> Stashed changes:src/utils.py

    def __read_vocab(self, input_path, return_data=False):
        with open(input_path, 'r') as content_file:
            data = content_file.read().split('\n')
        vocab_dict = {k: v for (v, k) in enumerate(data, start=1)}
        if return_data:
            return vocab_dict, data
        return vocab_dict

    def __read_data(self, input_path):
        with open(input_path, 'r') as input_file:
            data = input_file.readlines()
        return data

    def __helper(self, i, j):
        code_oovs = []
        nl_batch = []
        ast_batch = []
        code_batch = []
        ex_code_batch = []
        ex_nl_batch = []
<<<<<<< Updated upstream:src-tf/DataWorks.py
        maxlen = 670
=======
>>>>>>> Stashed changes:src/utils.py

        for line in self.__code_data[i:j]:
            idxs_ex = []
            line = f"<S> {line} </S>"
            for word in line.split():
                try:
                    idxs_ex.append(self.__code_vocab[word])
                except KeyError:
                    if word not in code_oovs:
                        code_oovs.append(word)
<<<<<<< Updated upstream:src-tf/DataWorks.py
                    idxs_ex.append(len(self.__code_vocab) +
                                   code_oovs.index(word) + 1)
            ex_code_batch.append(idxs_ex)
            code_batch.append(self.__fill_UNK(idxs_ex, self.__code_vocab))
        code_batch = pad_sequences(code_batch, padding="post")
        ex_code_batch = pad_sequences(ex_code_batch, padding="post")
=======
                    idxs_ex.append(len(self.__code_vocab) + code_oovs.index(word))
            ex_code_batch.append(idxs_ex)
            code_batch.append(self.__fill_UNK(idxs_ex, self.__code_vocab))
>>>>>>> Stashed changes:src/utils.py

        for line in self.__ast_data[i:j]:
            line = f"<S> {line} </S>"
            ast_batch.append([self.__ast_vocab[w] for w in line.split()])
<<<<<<< Updated upstream:src-tf/DataWorks.py
        ast_batch = pad_sequences(ast_batch, padding="post")
=======
>>>>>>> Stashed changes:src/utils.py

        for line in self.__nl_data[i:j]:
            idxs_ex = []
            line = f"<S> {line} </S>"
            for word in line.split():
                try:
                    idxs_ex.append(self.__nl_vocab[word])
                except KeyError:
                    try:
<<<<<<< Updated upstream:src-tf/DataWorks.py
                        idxs_ex.append(len(self.__nl_vocab) + 
                                       code_oovs.index(word) + 1)
=======
                        idxs_ex.append(len(self.__nl_vocab) + code_oovs.index(word)) 
>>>>>>> Stashed changes:src/utils.py
                    except ValueError:
                        idxs_ex.append(self.__nl_vocab["<UNK>"])
            ex_nl_batch.append(idxs_ex)
            nl_batch.append(self.__fill_UNK(idxs_ex, self.__nl_vocab))
<<<<<<< Updated upstream:src-tf/DataWorks.py
        nl_batch = pad_sequences(nl_batch, padding="post")
        ex_nl_batch = pad_sequences(ex_nl_batch, padding="post")
        
        ast_batch = pad_sequences(ast_batch, maxlen=maxlen, padding="post")
        code_batch = pad_sequences(code_batch, maxlen=maxlen, padding="post")
        ex_code_batch = pad_sequences(ex_code_batch, maxlen=maxlen, padding="post")
=======
        
        l = max(len(max(code_batch, key=len)), len(max(ast_batch, key=len)))
        nl_batch = pad_sequences(nl_batch, padding="post")
        ex_nl_batch = pad_sequences(ex_nl_batch, padding="post")
        code_batch = pad_sequences(code_batch, padding="post", maxlen=l)
        ast_batch = pad_sequences(ast_batch, padding="post", maxlen=l)
        ex_code_batch = pad_sequences(ex_code_batch, padding="post", maxlen=l)
>>>>>>> Stashed changes:src/utils.py

        batch = Batch()
        batch.nl = nl_batch
        batch.ast = ast_batch
        batch.code = code_batch
        batch.nl_ex = ex_nl_batch
        batch.code_oovs = code_oovs
        batch.code_ex = ex_code_batch
        batch.max_code_oovs = len(code_oovs)
        return batch

    def batcher(self, shuffle=True):
        if shuffle:
            combined = list(zip(self.__code_data, self.__ast_data, self.__nl_data))
            random.shuffle(combined)
            self.__code_data, self.__ast_data, self.__nl_data = zip(*combined)

        i, l = 0, len(self.__code_data)
        while i + self.batch_sz < l:
            j = i + self.batch_sz
            yield self.__helper(i, j)
            i = j