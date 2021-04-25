from time import time
import tensorflow as tf

class Trainer:
    def __init__(self, encoder, decoder, optimizer, batchqueue, batch_sz, epochs, logging, ckpt):
        self.ckpt = ckpt
        self.epochs = epochs
        self.encoder = encoder
        self.decoder = decoder
        self.logging = logging
        self.batch_sz = batch_sz
        self.optimizer = optimizer
        self.batchqueue = batchqueue.batcher(self.batch_sz)

    def __train_step(self, code, code_ext, ast, target, max_oovs):
        with tf.GradientTape() as tape:
            total_loss, coverage = 0, None
            hidden, state_h, state_c = self.encoder(code, ast)
            # initially feed in batch_sz tensor of start tokens <S>
            # for us start token is always first word in the vocab
            dec_input = tf.expand_dims([1] * self.batch_sz, axis=1)
            
            step_losses = []
            for i in range(1, target.shape[1]):
                final_dist, attn_dist, state_h, state_c, coverage = self.decoder(dec_input, hidden,
                                                                                 prev_h=state_h,
                                                                                 prev_c=state_c,
                                                                                 coverage=coverage,
                                                                                 max_oovs=max_oovs,
                                                                                 inp_code_ext=code_ext)
                                                                                     
                actual_word = tf.expand_dims(target[:,i], axis=1) #teacher forcing
                gold_probs = tf.squeeze(tf.gather_nd(final_dist, actual_word, batch_dims=1))
                cov_loss = tf.reduce_sum(tf.minimum(coverage, attn_dist), axis=1)
                step_loss = -tf.math.log(gold_probs + 1e-12) + cov_loss
                step_losses.append(step_loss)
                # output at t-1 timestep is input for t timestep
                dec_input = actual_word
                
            sum_losses = tf.reduce_sum(tf.stack(step_losses, axis=1), axis=1)
            batch_avg_loss = sum_losses/target.shape[1]
            final_loss = tf.reduce_mean(batch_avg_loss)
                            
            variables = self.encoder.trainable_variables + \
                        self.decoder.trainable_variables

            grads = tape.gradient(final_loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return final_loss.numpy()

    def train(self):
        avg_time_per_batch = 0
        for epoch in range(self.epochs):
            print(f"[INFO] Running epoch: {epoch}")
            for (nbatch, batch) in enumerate(self.batchqueue):
                start = time()
                loss = self.__train_step(batch.code, 
                                         batch.code_ex, 
                                         batch.ast, 
                                         batch.nl, 
                                         batch.max_oovs)
                
                avg_time_per_batch += (time() - start)
                if not epoch and not nbatch:
                    self.encoder.summary()
                    self.decoder.summary()
                if not nbatch % self.logging:
                    avg_time_per_batch /= self.logging
                    print("[INFO] Batch: {} | Loss: {:.2f} | {:.2f} sec/batch".format(nbatch, loss, avg_time_per_batch))
                    avg_time_per_batch = 0

    def store_checkpoint():
        pass

    def retrieve_checkpoint():
        pass