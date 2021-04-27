import config
import argparse
from Trainer import Trainer
from utils import BatchQueue
from tensorflow.keras import mixed_precision
from models import DeepCom, AttentionDecoder
from tensorflow_addons.optimizers import Lookahead, SGDW
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
mixed_precision.set_global_policy('mixed_float16')


parser = argparse.ArgumentParser(description="Run the Model")
parser.add_argument("-b", "--batch-size", dest="bs", type=int,
                    default=8, help="Batch size for the model")
parser.add_argument("-e", "--epochs", type=int, default=10,
                    help="Number of epochs for the model")
parser.add_argument("-lr", "--learning-rate", dest="lr",
                    type=float, default=1e-3, help="learning rate for the model")
parser.add_argument("-o", "--optimizer", type=str,
                    default="adagrad", help="type of optimizer")
parser.add_argument("-log", "--logging", type=int, dest="logging",
                    default=100, help="log the loss after x batches")
parser.add_argument("-la", "--look-ahead", dest="la", action="store_true")
parser.add_argument("-ckpt", "--check-point-after", dest="ckpt",
                    default=250, help="check point the model after these many batches")
args = parser.parse_args()


avail_optims = {
    "rmsprop": RMSprop(learning_rate=args.lr),
    "adagrad": Adagrad(learning_rate=args.lr),
    "adam": Adam(learning_rate=args.lr, amsgrad=True),
    "sgd": SGD(learning_rate=args.lr, momentum=0.9, nesterov=True),
    "sgdw": SGDW(weight_decay=1e-5, learning_rate=args.lr, momentum=0.9, nesterov=True)
}


batchqueue = BatchQueue(args.bs, key="train")

encoder = DeepCom(inp_dim_code=config.vocab_size_code+1,
                  inp_dim_ast=config.vocab_size_ast+1)

decoder = AttentionDecoder(inp_dim=config.vocab_size_nl+1)

print(f"[INFO] Using the optimizer {args.optimizer}")
optim = avail_optims[args.optimizer]

if args.la:
    print("[INFO] Using LookAhead wrapper on optimizer")
    optim = Lookahead(optim)
    
optim = mixed_precision.LossScaleOptimizer(optim)

model_trainer = Trainer(encoder=encoder,
                        decoder=decoder,
                        optimizer=optim,
                        batchqueue=batchqueue,
                        batch_sz=args.bs,
                        epochs=args.epochs,
                        logging=args.logging,
                        ckpt=args.ckpt)

model_trainer.train()
