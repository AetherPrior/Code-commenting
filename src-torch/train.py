import torch
import config
import argparse
from UtilClasses import Trainer
from DataWorks import BatchQueue
from torch_optimizer import Ranger
from torch.cuda import is_available
from torch.optim.lr_scheduler import ExponentialLR
from Models import DeepComEncoder, AttentionDecoder
device = torch.device("cuda" if is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run the Model")
parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size for the model")
parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs for the model")
parser.add_argument("-lr", "--learning-rate", type=float, default=0.01, help="learning rate for the model")
parser.add_argument("-log", "--logging", type=int, default=1, help="log the loss after `X` batches")
parser.add_argument("-ckpt", "--check-point-after", type=int, default=1700, help="check point the model after `X` batches")
parser.add_argument("-cov", "--coverage", type=float, default=1.0, help="set coverage lambda parameter")
args = parser.parse_args()
                                      
batchqueue = BatchQueue(args.batch_size, key="train")

encoder = DeepComEncoder(inp_dim_code=config.vocab_size_code+1,
                         inp_dim_ast=config.vocab_size_ast+1).to(device)

decoder = AttentionDecoder(inp_dim=config.vocab_size_nl+1).to(device)

params = list(encoder.parameters()) + list(decoder.parameters())
optim = Ranger(params, lr=args.learning_rate, weight_decay=1e-5)
sched = ExponentialLR(optim, gamma=0.99, last_epoch=-1)
    
model_trainer = Trainer(encoder=encoder,
                        decoder=decoder,
                        optimizer=optim,
                        scheduler=sched,
                        batchqueue=batchqueue,
                        batch_sz=args.batch_size,
                        epochs=args.epochs,
                        logging=args.logging,
                        ckpt=args.check_point_after,
                        cov=args.coverage)

model_trainer.train()