from config import *


class VocabData:
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as content_file:
            data = content_file.read().split('\n')
            content_file.close()
        self.vocab_dict = dict([(y, x) for (x, y) in enumerate(data)])

    def add_pointer(self, sentence):
        '''
        tokenize the sentence
        then add their words to the vocab dict
        '''
        pass
