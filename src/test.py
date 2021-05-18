import config
import argparse
from UtilClasses import Tester
from DataWorks import BatchQueue
from Models import DeepComEncoder, AttentionDecoder


parser = argparse.ArgumentParser(description="Run the Model")
parser.add_argument("-b", "--batch-size", dest="bs", type=int,
                    default=128, help="Batch size for the model")
parser.add_argument("-l2f", "--log-to-file", dest="l2f", action="store_true")
args = parser.parse_args()


batchqueue = BatchQueue(args.bs, key="test")

encoder = DeepComEncoder(inp_dim_code=config.vocab_size_code+1,
                         inp_dim_ast=config.vocab_size_ast+1)

decoder = AttentionDecoder(inp_dim=config.vocab_size_nl+1)

model_tester = Tester(encoder=encoder,
                      decoder=decoder,
                      batchqueue=batchqueue,
                      batch_sz=args.bs)

model_tester.test(args.l2f)
