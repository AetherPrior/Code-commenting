import config
from time import time
import tensorflow as tf
from tqdm import tqdm
from os.path import exists
from tensorflow.train import Checkpoint
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#---------------------------------#
import nltk
nltk.download("wordnet")
#---------------------------------#

class Trainer:
    def __init__(self, encoder, decoder, optimizer, batchqueue, batch_sz, epochs, logging, ckpt, cov=1.0):
        self.ckpt_num = ckpt
        self.epochs = epochs
        self.cov_lambda = cov
        self.encoder = encoder
        self.decoder = decoder
        self.logging = logging
        self.batch_sz = batch_sz
        self.optimizer = optimizer
        self.batchqueue = batchqueue
        self.checkpoint = Checkpoint(encoder=self.encoder,
                                     decoder=self.decoder,
                                     optimizer=self.optimizer)
                   
    def __train_step(self, batch):
        with tf.GradientTape() as tape:
            total_loss, coverage, context_vector = 0, None, None
            hidden, state_h, state_c = self.encoder(batch.code, batch.ast)
            # initially feed in batch_sz tensor of start tokens <S>
            # for us start token is always first word in the vocab
            dec_input = tf.expand_dims([config.BOS] * self.batch_sz, axis=1)

            step_losses, comment_batch = [], []
            state = [state_h, state_c]
            for i in range(1, batch.nl.shape[1]):
                final_dist, attn_dist, state, coverage, context_vector = self.decoder(dec_input, hidden,
                                                                                      prev_state=state,
                                                                                      coverage=coverage,
                                                                                      context_vector=context_vector,
                                                                                      code_ext=batch.code_ex,
                                                                                      max_oovs=len(batch.code_oovs))

                comment_batch.append(tf.math.argmax(final_dist, axis=1))
                actual_word = tf.expand_dims(batch.nl_ex[:, i], axis=1)
                gold_probs = tf.squeeze(tf.gather_nd(final_dist, actual_word, batch_dims=1))
                cov_loss = tf.reduce_sum(tf.minimum(coverage, attn_dist), axis=1)
                step_loss = -tf.math.log(gold_probs + 1e-12) + (cov_loss * self.cov_lambda)
                dec_input = tf.expand_dims(batch.nl[:, i], axis=1) # teacher forcing
                step_losses.append(step_loss)

            sum_losses = tf.reduce_sum(tf.stack(step_losses, axis=1), axis=1)
            batch_avg_loss = sum_losses/batch.nl.shape[1]
            final_loss = tf.reduce_mean(batch_avg_loss)

        variables = self.encoder.trainable_variables + \
                    self.decoder.trainable_variables

        grads = tape.gradient(final_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        comment_batch = tf.transpose(tf.stack(comment_batch))
        return final_loss.numpy(), comment_batch


    def train(self, log_to_file=False):
        if log_to_file:
            logfile = open("log_train.txt", 'a')
        
        self.__retrieve_checkpoint()
        for epoch in range(self.epochs):
            temp_batchqueue = self.batchqueue.batcher(self.batch_sz)
            self.nl_list = self.batchqueue.nl_list
            
            for (nbatch, batch) in enumerate(tqdm(temp_batchqueue)):
                target_comment = batch.nl[:, 1:]
                loss, comment_batch = self.__train_step(batch)
                out = f"\n[INFO] Step: [{epoch}, {nbatch}] | Loss: {loss:.2f}\n"
                out += self.__get_result(comment_batch.numpy(), 
                                         target_comment, 
                                         batch.code_oovs)
                if log_to_file:
                    logfile.write(f"{out}\n")
                    logfile.flush()
                if not nbatch % self.logging:
                    print(out, flush=True)
                if self.ckpt_num and nbatch and not nbatch % self.ckpt_num:
                    self.checkpoint.write("./ckpts/ckpt")
                    print("\n[INFO] Stored checkpoint")
                    

    def __retrieve_checkpoint(self):
        if exists("./ckpts/ckpt.index"):
            try:
                self.checkpoint.read("./ckpts/ckpt").expect_partial()
                print("[INFO] Read Checkpoint")
            except:
                print("[INFO] Failed reading Checkpoint, going with no checkpoint")
        else:
            print("[INFO] Starting from scratch")
            
    
    def __trim_string(self, s):
        trimmed = []
        for w in s:
            if w == "</S>" or w == "<PAD>":
                break
            trimmed.append(w)
        if not len(trimmed):
            trimmed.append("<PAD>")
            trimmed.append("<PAD>") # method4 requires hyp to be >1
        return trimmed


    def __get_result(self, comment_batch, target_comment, oovs, print_idx=None):
        """
        BLEU Scoring
        """
        batch_len = len(comment_batch)
        BLEU4_score, METEOR_score = 0, 0
        
        for i in range(batch_len):
            target = ["<PAD>" if not j else self.nl_list[j-1] for j in target_comment[i]]
            
            comment_hat = []
            for word in comment_batch[i]:
                if not word:
                    comment_hat.append("<PAD>")
                elif word <= config.vocab_size_nl :
                    comment_hat.append(self.nl_list[word-1])
                else:
                    comment_hat.append(oovs[word-1-config.vocab_size_nl])
            
            target = self.__trim_string(target)
            comment_hat = self.__trim_string(comment_hat)
            
            if print_idx is not None and print_idx == i:
                print(f"Prediction: {' '.join(comment_hat)}")
                print(f"Ground Truth: {' '.join(target)}", flush=True)
            
            smoothie = SmoothingFunction().method4
            BLEU4_score += sentence_bleu([target], 
                                         comment_hat, 
                                         auto_reweigh=True,
                                         smoothing_function=smoothie) * 100
                                         
            METEOR_score += meteor_score([' '.join(target)],
                                         ' '.join(comment_hat)) * 100
   
        return f"Cumulative BLEU4 score: {(BLEU4_score/batch_len):.2f}\n" + \
               f"Cumulative METEOR score: {(METEOR_score/batch_len):.2f}"
               
#-------------------------------------------------------------------------------------------------------#               
               
class Tester:
    def __init__(self, encoder, decoder, batchqueue, batch_sz):
        self.encoder = encoder
        self.decoder = decoder
        self.batch_sz = batch_sz
        self.batchqueue = batchqueue
        self.checkpoint = Checkpoint(encoder=self.encoder,
                                     decoder=self.decoder)
                                              

    def __test_step(self, batch):
        comment_batch, coverage, context_vector = [], None, None
        hidden, state_h, state_c = self.encoder(batch.code, batch.ast)
        # initially feed in batch_sz tensor of start tokens <S>
        # for us start token is always first word in the vocab
        dec_input = tf.expand_dims([config.BOS] * self.batch_sz, axis=1)
        
        state = [state_h, state_c]
        for i in range(1, batch.nl.shape[1]):
            final_dist, _, state, coverage, context_vector = self.decoder(dec_input, hidden,
                                                                          prev_state=state,
                                                                          coverage=coverage,
                                                                          context_vector=context_vector,
                                                                          code_ext=batch.code_ex,
                                                                          max_oovs=len(batch.code_oovs))
            
            max_prob_words = tf.math.argmax(final_dist, axis=1).numpy()
            comment_batch.append(max_prob_words)
            
            max_prob_words = [config.UNK if w >= config.vocab_size_nl else w for w in max_prob_words]
            dec_input = tf.expand_dims(max_prob_words, axis=1)
            
        comment_batch = tf.transpose(tf.stack(comment_batch, axis=0))
        return comment_batch


    def test(self, log_to_file=False):
        if log_to_file:
            logfile = open("log_test.txt", 'a')
        
        self.__retrieve_checkpoint()
        temp_batchqueue = self.batchqueue.batcher(self.batch_sz)
        self.nl_list = self.batchqueue.nl_list
            
        for batch in tqdm(temp_batchqueue, ncols=100):
            target_comment = batch.nl[:, 1:]
            comment_batch = self.__test_step(batch)
            out = self.__get_result(comment_batch.numpy(), 
                                    target_comment, 
                                    batch.code_oovs,
                                    print_idx=10)
            if log_to_file:
                logfile.write(f"{out}\n")
                logfile.flush()
            print(out, flush=True)


    def __retrieve_checkpoint(self):
        if exists("./ckpts/ckpt.index"):
            try:
                self.checkpoint.read("./ckpts/ckpt").expect_partial()
                print("[INFO] Read Checkpoint")
            except:
                print("[INFO] Failed reading Checkpoint. Please rectify errors and try again")
        else:
            print("[INFO] Model trained weights are not available. Please train the model first and then test.")
            
    
    def __trim_string(self, s):
        trimmed = []
        for w in s:
            if w == "</S>" or w == "<PAD>":
                break
            trimmed.append(w)
        if not len(trimmed):
            trimmed.append("<PAD>")
            trimmed.append("<PAD>") # method4 requires hyp to be >1
        return trimmed


    def __get_result(self, comment_batch, target_comment, oovs, print_idx=None):
        """
        BLEU Scoring
        """
        batch_len = len(comment_batch)
        BLEU4_score, METEOR_score = 0, 0
        
        for i in range(batch_len):
            target = ["<PAD>" if not j else self.nl_list[j-1] for j in target_comment[i]]
            
            comment_hat = []
            for word in comment_batch[i]:
                if not word:
                    comment_hat.append("<PAD>")
                elif word <= config.vocab_size_nl :
                    comment_hat.append(self.nl_list[word-1])
                else:
                    comment_hat.append(oovs[word-1-config.vocab_size_nl])
            
            target = self.__trim_string(target)
            comment_hat = self.__trim_string(comment_hat)
            
            if print_idx is not None and print_idx == i:
                print(f"Prediction: {' '.join(comment_hat)}")
                print(f"Ground Truth: {' '.join(target)}", flush=True)
            
            smoothie = SmoothingFunction().method4
            BLEU4_score += sentence_bleu([target], 
                                         comment_hat, 
                                         auto_reweigh=True,
                                         smoothing_function=smoothie) * 100
                                         
            METEOR_score += meteor_score([' '.join(target)], 
                                         ' '.join(comment_hat)) * 100
   
        return f"Cumulative BLEU4 score: {(BLEU4_score/batch_len):.2f}\n" + \
               f"Cumulative METEOR score: {(METEOR_score/batch_len):.2f}\n"