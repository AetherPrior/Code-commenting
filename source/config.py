from os.path import join

ROOT_VOCAB = "../Dataset/data_RQ1/"
AST_VOCAB = join(ROOT_VOCAB, "vocab.ast")
CODE_VOCAB = join(ROOT_VOCAB, "vocab.code")
NL_VOCAB = join(ROOT_VOCAB, "vocab.nl")

CODE_INPUT = join(ROOT_VOCAB, "train/train.token.code")
AST_INPUT = join(ROOT_VOCAB, "train/train.token.ast")
NL_INPUT = join(ROOT_VOCAB, "train/train.token.nl")

PAD = 0
BOS = 1
EOS = 2
UNK = 3

vocab_size_code = 30000
vocab_size_nl = 30000
vocab_size_ast = 65

