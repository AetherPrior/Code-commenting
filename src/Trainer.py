import sys
import tensorflow as tf
from os.path import exists

class Trainer:
    def __init__(self, encoder, decoder, optimizer, batchqueue, batch_sz, epochs, logging, ckpt):
        self.ckpt_num = ckpt
        self.epochs = epochs
        self.encoder = encoder
        self.decoder = decoder
        self.logging = logging
        self.batch_sz = batch_sz
        self.optimizer = optimizer
        self.batchqueue = batchqueue
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder, 
                                              decoder=self.decoder, 
                                              step=tf.Variable(0), 
                                              optimizer=self.optimizer)

    def __train_step(self, batch):
        with tf.GradientTape() as tape:
            total_loss, coverage = 0, None
            hidden, state_h, state_c = self.encoder(batch.code, batch.ast)
            # initially feed in batch_sz tensor of start tokens <S>
            # for us start token is always first word in the vocab
            dec_input = tf.expand_dims([1] * self.batch_sz, axis=1)
            
            step_losses = []
            for i in range(1, batch.nl.shape[1]):
                final_dist, attn_dist, state_h, state_c, coverage = self.decoder(dec_input, hidden,
                                                                                 prev_h=state_h,
                                                                                 prev_c=state_c,
                                                                                 coverage=coverage,
                                                                                 max_oovs=batch.max_oovs,
                                                                                 code_ext=batch.code_ex)
                                                                                     
                actual_word = tf.expand_dims(batch.nl[:,i], axis=1) #teacher forcing
                gold_probs = tf.squeeze(tf.gather_nd(final_dist, actual_word, batch_dims=1))
                cov_loss = tf.reduce_sum(tf.minimum(coverage, attn_dist), axis=1)
                step_loss = -tf.math.log(gold_probs + 1e-12) + cov_loss
                step_losses.append(step_loss)
                dec_input = actual_word
                
            sum_losses = tf.reduce_sum(tf.stack(step_losses, axis=1), axis=1)
            batch_avg_loss = sum_losses/batch.nl.shape[1]
            final_loss = tf.reduce_mean(batch_avg_loss)
                            
            variables = self.encoder.trainable_variables + \
                        self.decoder.trainable_variables

            grads = tape.gradient(final_loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return final_loss.numpy()

    def train(self):
        self.retrieve_checkpoint()
                
        for epoch in range(self.epochs):
            print(f"[INFO] Running epoch: {epoch}")
            temp_batchqueue = self.batchqueue.batcher(self.batch_sz)
            for (nbatch, batch) in enumerate(temp_batchqueue):
                loss = self.__train_step(batch)
                
                if not epoch and not nbatch:
                    self.encoder.summary()
                    self.decoder.summary()
                if not nbatch % self.logging:
                    out = "[INFO] Batch: {} | Loss: {:.2f}".format(nbatch, loss)
                    with open("my_log.txt", 'a') as the_file:
                        the_file.write(out + '\n')
                    print(out)
                if self.ckpt_num and nbatch and not nbatch % self.ckpt_num:
                    self.checkpoint.step.assign_add(self.ckpt_num)
                    self.store_checkpoint()
                    print(f"[INFO] Stored checkpoint for step {int(self.checkpoint.step)}")

    def store_checkpoint(self):
        self.checkpoint.write("./ckpts/ckpt")

    def retrieve_checkpoint(self):
        if exists("./ckpts/ckpt.index"):
            try:
                self.checkpoint.read("./ckpts/ckpt")
                print("[INFO] Read Checkpoint")
            except:
                print("[INFO] Failed reading Checkpoint, going with no checkpoint")
                print(sys.exc_info()[0])
        else:
            print("[INFO] Starting from scratch")