import config
import argparse
from DataWorks import BatchQueue
from RangeTrainer import Trainer
from Models import DeepComEncoder, AttentionDecoder
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow_addons.optimizers import Lookahead, SGDW, AdamW, RectifiedAdam


parser = argparse.ArgumentParser(description="Run a range-test on the Model")
parser.add_argument("-b", "--batch-size", dest="bs", type=int,
                    default=128, help="Batch size for the model")
parser.add_argument("-e", "--epochs", type=int, default=5,
                    help="Number of epochs for the model")
parser.add_argument("-lr", "--learning-rate", dest="lr",
                    type=float, default=1e-2, help="learning rate for the model")
parser.add_argument("-o", "--optimizer", type=str,
                    default="radam", help="type of optimizer")
parser.add_argument("-log", "--logging", type=int, dest="logging",
                    default=1, help="log the loss after x batches")
parser.add_argument("-la", "--look-ahead", dest="la", action="store_true")
parser.add_argument("-cov", "--coverage", type=float, 
                    default=1.0, help="set coverage lambda parameter")
parser.add_argument("-l2f", "--log-to-file", dest="l2f", action="store_true")
parser.add_argument("-si", "--step-increment", type=float, default=1e-3, 
                    dest="si", help="step increment for learning rate")
args = parser.parse_args()

                                      
avail_optims = {
    "adagrad": Adagrad(learning_rate=args.lr),
    "adam": Adam(learning_rate=args.lr, amsgrad=True),
    "adamw": AdamW(weight_decay=1e-5, learning_rate=args.lr),
    "sgd": SGD(learning_rate=args.lr, momentum=0.9, nesterov=True),
    "sgdw": SGDW(weight_decay=1e-5, learning_rate=args.lr, momentum=0.9, nesterov=True),
    "radam": RectifiedAdam(learning_rate=args.lr, weight_decay=1e-5, min_lr=1e-5, amsgrad=True)
}

batchqueue = BatchQueue(args.bs, key="train")

encoder = DeepComEncoder(inp_dim_code=config.vocab_size_code+1,
                         inp_dim_ast=config.vocab_size_ast+1)

decoder = AttentionDecoder(inp_dim=config.vocab_size_nl+1)

print(f"[INFO] Using the optimizer {args.optimizer}")
optim = avail_optims[args.optimizer]

if args.la:
    print(f"[INFO] Using Lookahead wrapper on {args.optimizer}")
    optim = Lookahead(optim)
    
print("[INFO] Starting Range Training")
    
model_trainer = Trainer(encoder=encoder,
                        decoder=decoder,
                        optimizer=optim,
                        batchqueue=batchqueue,
                        batch_sz=args.bs,
                        epochs=args.epochs,
                        logging=args.logging,
                        cov=args.coverage,
                        step_inc=args.si)

model_trainer.train(args.l2f)
