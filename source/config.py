from os.path import join

ROOT_VOCAB = "./Dataset/data_RQ1/"
AST_VOCAB = join(ROOT_VOCAB, "vocab.ast")
CODE_VOCAB = join(ROOT_VOCAB, "vocab.code")
NL_VOCAB = join(ROOT_VOCAB, "vocab.nl")

PAD = 0
UNK = 1
BOS = 2
EOS = 3
