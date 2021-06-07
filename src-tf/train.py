import config
import argparse
from UtilClasses import Trainer
from DataWorks import BatchQueue
from Models import DeepComEncoder, AttentionDecoder
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow_addons.optimizers import Lookahead, SGDW, AdamW, RectifiedAdam


parser = argparse.ArgumentParser(description="Run the Model")
parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size for the model")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for the model")
parser.add_argument("-lr", "--learning-rate", type=float, default=0.015, help="learning rate for the model")
parser.add_argument("-o", "--optimizer", type=str, default="radam", help="type of optimizer")
parser.add_argument("-log", "--logging", type=int, default=100, help="log the loss after `X` batches")
parser.add_argument("-la", "--look-ahead", action="store_true")
parser.add_argument("-ckpt", "--check-point-after", type=int, default=0, help="check point the model after `X` batches")
parser.add_argument("-cov", "--coverage", type=float, default=1.0, help="set coverage lambda parameter")
parser.add_argument("-l2f", "--log-to-file", action="store_true")
args = parser.parse_args()


lr_schedule = ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=3482, decay_rate=0.95)
                                      
avail_optims = {
    "adam": Adam(learning_rate=lr_schedule, amsgrad=True, clipnorm=5.0),
    "adamw": AdamW(weight_decay=1e-5, learning_rate=lr_schedule, clipnorm=5.0),
    "sgd": SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True, clipnorm=5.0),
    "sgdw": SGDW(weight_decay=1e-5, learning_rate=lr_schedule, momentum=0.9, nesterov=True, clipnorm=5.0),
    "radam": RectifiedAdam(learning_rate=lr_schedule, weight_decay=1e-5, amsgrad=True, clipnorm=5.0)
}

batchqueue = BatchQueue(args.batch_size, key="train")

encoder = DeepComEncoder(inp_dim_code=config.vocab_size_code+1,
                         inp_dim_ast=config.vocab_size_ast+1) # 30001, AST_VOCAB + 1

decoder = AttentionDecoder(inp_dim=config.vocab_size_nl+1) #30001

print(f"[INFO] Using the optimizer {args.optimizer}")
optim = avail_optims[args.optimizer]

if args.look_ahead:
    print(f"[INFO] Using Lookahead wrapper on {args.optimizer}")
    optim = Lookahead(optim)
    
model_trainer = Trainer(encoder=encoder,
                        decoder=decoder,
                        optimizer=optim,
                        batchqueue=batchqueue,
                        batch_sz=args.batch_size,
                        epochs=args.epochs,
                        logging=args.logging,
                        ckpt=args.check_point_after,
                        cov=args.coverage)

model_trainer.train(args.log_to_file)
