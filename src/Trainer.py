import sys
import numpy as np
import tensorflow as tf
from os.path import exists
from random import randint
from tensorflow.train import Checkpoint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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
        self.cov_lambda = cov
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                              decoder=self.decoder,
                                              optimizer=self.optimizer)
                                              

    def __train_step(self, batch):
        with tf.GradientTape() as gtape:
            total_loss, coverage = 0, None
            hidden, state_h, state_c = self.encoder(batch.code, batch.ast)
            # initially feed in batch_sz tensor of start tokens <S>
            # for us start token is always first word in the vocab
            dec_input = tf.expand_dims([1] * self.batch_sz, axis=1)

            step_losses = []
            comment_batch = []
            for i in range(1, batch.nl.shape[1]):
                final_dist, attn_dist, state_h, state_c, coverage = self.decoder(dec_input, hidden,
                                                                                 prev_h=state_h,
                                                                                 prev_c=state_c,
                                                                                 coverage=coverage,
                                                                                 max_oovs=batch.max_code_oovs,
                                                                                 code_ext=batch.code_ex)
                comment_batch.append(final_dist)
                actual_word = tf.expand_dims(
                    batch.nl[:, i], axis=1)  # teacher forcing
                gold_probs = tf.squeeze(tf.gather_nd(
                    final_dist, actual_word, batch_dims=1))
                cov_loss = tf.reduce_sum(
                    tf.minimum(coverage, attn_dist), axis=1)
                step_loss = -tf.math.log(gold_probs + 1e-12) + \
                    (cov_loss * self.cov_lambda)
                step_losses.append(step_loss)
                dec_input = actual_word

            sum_losses = tf.reduce_sum(tf.stack(step_losses, axis=1), axis=1)
            batch_avg_loss = sum_losses/batch.nl.shape[1]
            final_loss = tf.reduce_mean(batch_avg_loss)

        variables = self.encoder.trainable_variables + \
                    self.decoder.trainable_variables

        grads = gtape.gradient(final_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        comment_batch = np.argmax(np.array(comment_batch), axis=2).T
        return final_loss.numpy(), comment_batch, batch.nl[:, 1:]


    def train(self):
        self.__retrieve_checkpoint()
        for epoch in range(self.epochs):
            print(f"[INFO] Running epoch: {epoch}")
            temp_batchqueue = self.batchqueue.batcher(self.batch_sz)
            self.nl_list = self.batchqueue.nl_list
            
            for (nbatch, batch) in enumerate(temp_batchqueue):
                loss, comment_batch, target_comment = self.__train_step(batch)
                if not epoch and not nbatch:
                    self.encoder.summary()
                    self.decoder.summary()
                if not nbatch % self.logging:
                    print(f"\n[INFO] Batch: {nbatch} | Loss: {loss:.2f}")
                    self.__get_result(comment_batch, target_comment, batch.code_oovs)
                if self.ckpt_num and nbatch and not nbatch % self.ckpt_num:
                    self.checkpoint.write("./ckpts/ckpt")
                    print("\n[INFO] Stored checkpoint\n")


    def __retrieve_checkpoint(self):
        if exists("./ckpts/ckpt.index"):
            try:
                self.checkpoint.read("./ckpts/ckpt").expect_partial()
                print("[INFO] Read Checkpoint")
            except:
                print("[INFO] Failed reading Checkpoint, going with no checkpoint")
                print(sys.exc_info()[0])
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
        return trimmed


    def __get_result(self, comment_batch, target_comment, oovs, print_batch=None):
        """
        BLEU Scoring
        """
        batch_len = len(comment_batch)
        mean_BLEU_1, mean_BLEU_2, mean_BLEU_3, mean_BLEU_4 = 0, 0, 0, 0
        
        for comment in range(0, batch_len):
            
            comment_hat = []
            target = ["<PAD>" if not j else self.nl_list[j-1] for j in target_comment[comment]]
            
            for word in comment_batch[comment]:
                if not word:
                    comment_hat.append("<PAD>")
                elif word < 29999:
                    comment_hat.append(self.nl_list[word-1])
                else:
                    comment_hat.append(oovs[word-29999])
            
            target = self.__trim_string(target)
            comment_hat = self.__trim_string(comment_hat)
            
            if print_batch is not None and print_batch == comment:
                print("Prediction:", " ".join(comment_hat))
                print("Ground Truth:", " ".join(target))
            
            smoothie = SmoothingFunction().method4
            mean_BLEU_1 += sentence_bleu([target], 
                                         comment_hat, 
                                         weights=(1, 0, 0, 0), 
                                         smoothing_function=smoothie)
            mean_BLEU_2 += sentence_bleu([target], 
                                         comment_hat, 
                                         weights=(0.5, 0.5, 0, 0),
                                         smoothing_function=smoothie)
            mean_BLEU_3 += sentence_bleu([target], 
                                         comment_hat, 
                                         weights=(0.33, 0.33, 0.33, 0),
                                         smoothing_function=smoothie)
            mean_BLEU_4 += sentence_bleu([target], 
                                         comment_hat, 
                                         weights=(0.25, 0.25, 0.25, 0.25),
                                         smoothing_function=smoothie)
            
        print(f"Cumulative 1-gram: {mean_BLEU_1/batch_len}")
        print(f"Cumulative 2-gram: {mean_BLEU_2/batch_len}")
        print(f"Cumulative 3-gram: {mean_BLEU_3/batch_len}")
        print(f"Cumulative 4-gram: {mean_BLEU_4/batch_len}")
        
