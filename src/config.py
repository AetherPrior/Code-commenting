from os.path import join

ROOT_VOCAB = "../Dataset/data_RQ1/"
AST_VOCAB = join(ROOT_VOCAB, "vocab.ast")
CODE_VOCAB = join(ROOT_VOCAB, "vocab.code")
NL_VOCAB = join(ROOT_VOCAB, "vocab.nl")

vocab_size_code = 30000
vocab_size_nl = 30000
vocab_size_ast = 65

paths = {

    "train": {
        "CODE_INPUT": join(ROOT_VOCAB, "train/train.token.code"),
        "AST_INPUT": join(ROOT_VOCAB, "train/train.token.ast"),
        "NL_INPUT" : join(ROOT_VOCAB, "train/train.token.nl")
    },

    "valid": {
        "CODE_INPUT": join(ROOT_VOCAB, "valid/valid.token.code"),
        "AST_INPUT": join(ROOT_VOCAB, "valid/valid.token.ast"),
        "NL_INPUT" : join(ROOT_VOCAB, "valid/valid.token.nl")
    },

    "test": {
        "CODE_INPUT": join(ROOT_VOCAB, "test/test.token.code"),
        "AST_INPUT": join(ROOT_VOCAB, "test/test.token.ast"),
        "NL_INPUT" : join(ROOT_VOCAB, "test/test.token.nl")
    }
}