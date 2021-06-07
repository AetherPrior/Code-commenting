import config
from time import time
import tensorflow as tf


class Trainer:
    def __init__(self, encoder, decoder, optimizer, batchqueue, batch_sz, epochs, logging, cov, step_inc):
        self.epochs = epochs
        self.cov_lambda = cov
        self.encoder = encoder
        self.decoder = decoder
        self.logging = logging
        self.batch_sz = batch_sz
        self.step_inc = step_inc
        self.optimizer = optimizer
        self.batchqueue = batchqueue
                                              

    def __train_step(self, batch):
        with tf.GradientTape() as tape:
            self.decoder.prev_context_vector = None
            total_loss, coverage = 0, None
            hidden, state_h, state_c = self.encoder(batch.code, batch.ast)
            # initially feed in batch_sz tensor of start tokens <S>
            # for us start token is always first word in the vocab
            dec_input = tf.expand_dims([config.BOS] * self.batch_sz, axis=1)

            step_losses = []
            state = [state_h, state_c]
            for i in range(1, batch.nl.shape[1]):
                final_dist, attn_dist, state, coverage = self.decoder(dec_input, hidden,
                                                                      prev_state=state,
                                                                      coverage=coverage,
                                                                      code_ext=batch.code_ex,
                                                                      max_oovs=batch.max_code_oovs)

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
        return final_loss.numpy()


    def train(self, log_to_file=False):
        if log_to_file:
            logfile = open("log_range_train.txt", 'a')
        
        for epoch in range(self.epochs):
            start_time = time()
            temp_batchqueue = self.batchqueue.batcher(self.batch_sz)
            for (nbatch, batch) in enumerate(temp_batchqueue):
                lr = self.optimizer.learning_rate.numpy()
                loss = self.__train_step(batch)
                if not nbatch % self.logging:
                    out = f"[INFO] Step: [{epoch}, {nbatch}] | lr: {lr} | Loss: {loss}"
                    print(out)
                    if log_to_file:
                        logfile.write(f"{out}\n")
                        # logfile.flush()
                self.optimizer.learning_rate.assign(lr + self.step_inc)
            print(f"Time to complete epoch {(time() - start_time)/60.0:.2f} min")