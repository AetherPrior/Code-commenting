import config
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Batch:
    def __init__(self):
        self.nl = None
        self.ast = None
        self.code = None
        self.oovs = None
        self.code_ex = None
        self.max_oovs = None


class BatchQueue:
    def __init__(self, batch_sz, key="train"):
        self.key = key
        self.batch_sz = batch_sz

    def __read_vocab(self, input_path):
        with open(input_path, 'r') as content_file:
            data = content_file.read().split('\n')
        return {k: v for (v, k) in enumerate(data, start=1)}

    def __read_data(self, input_path):
        with open(input_path, 'r') as input_file:
            data = input_file.readlines()
        return data

    def __helper(self, i, j):
            oovs = []
            nl_batch = []
            ast_batch = []
            code_batch = [] 
            ex_code_batch = []
            maxlen = 675
            
            for line in self.__code_data[i:j]:
                idxs, idxs_ex = [], []
                line = f"<S> {line} </S>"
                for word in line.split():
                    try:
                        idxs.append(self.__code_vocab[word])
                        idxs_ex.append(self.__code_vocab[word])
                    except KeyError:
                        if word not in oovs:
                            oovs.append(word)
                        idxs.append(self.__code_vocab["<UNK>"])
                        idxs_ex.append(len(self.__code_vocab) + oovs.index(word))
                code_batch.append(idxs)
                ex_code_batch.append(idxs_ex)
            code_batch = pad_sequences(code_batch, padding="post")
            ex_code_batch = pad_sequences(ex_code_batch, padding="post")

            for line in self.__ast_data[i:j]:
                idxs = []
                line = f"<S> {line} </S>"
                for word in line.split():
                    try:
                        idxs.append(self.__ast_vocab[word])
                    except KeyError:
                        idxs.append(self.__ast_vocab["<UNK>"])
                ast_batch.append(idxs)
            ast_batch = pad_sequences(ast_batch, padding="post")

            for line in self.__nl_data[i:j]:
                idxs = []
                line = f"<S> {line} </S>"
                for word in line.split():
                    try:
                        idxs.append(self.__nl_vocab[word])
                    except KeyError:
                        idxs.append(self.__nl_vocab["<UNK>"])
                nl_batch.append(idxs)
            nl_batch = pad_sequences(nl_batch, padding="post")
                    
            ast_batch = pad_sequences(ast_batch, maxlen=maxlen, padding="post")
            code_batch = pad_sequences(code_batch, maxlen=maxlen, padding="post")
            ex_code_batch = pad_sequences(code_batch, maxlen=maxlen, padding="post")

            batch = Batch()
            batch.oovs = oovs
            batch.nl = nl_batch
            batch.ast = ast_batch
            batch.code = code_batch
            batch.max_oovs = len(oovs)
            batch.code_ex = ex_code_batch
            return batch

    def batcher(self, shuffle=True):
        self.__nl_vocab = self.__read_vocab(config.NL_VOCAB)
        self.__ast_vocab = self.__read_vocab(config.AST_VOCAB)
        self.__code_vocab = self.__read_vocab(config.CODE_VOCAB)

        self.__nl_data = self.__read_data(config.paths[self.key]["NL_INPUT"])
        self.__ast_data = self.__read_data(config.paths[self.key]["AST_INPUT"])
        self.__code_data = self.__read_data(config.paths[self.key]["CODE_INPUT"])
        
        print(f"[INFO] Steps per epoch: {len(self.__code_data)//self.batch_sz}")

        if shuffle:
            combined = list(zip(self.__code_data, self.__ast_data, self.__nl_data))
            random.shuffle(combined)
            self.__code_data, self.__ast_data, self.__nl_data = zip(*combined)

        # here batcher can be extended to return
        # the shuffle dataset as it is
        i, l = 0, len(self.__code_data)
        while i + self.batch_sz < l:
            j = i + self.batch_sz
            yield self.__helper(i, j)
            i = j