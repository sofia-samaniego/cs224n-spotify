import re
import sys
import tensorflow as tf
import numpy as np
import time
import os

from model_nmt import Model
from data_util import load_data, load_and_preprocess_data, UNK_TOKEN, START_TOKEN, END_TOKEN, PAD_TOKEN
from util import Progbar, minibatches, padded_batch, tokens_to_sentences, padded_batch_lr
from rouge import rouge_n
from tensorflow.python.layers import core as layers_core
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from datetime import datetime, date
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

logger = logging.getLogger("project.milestone")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

global UNK_IDX, START_IDX, END_IDX, PAD_IDX
#debug = False
debug = True

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    Things to add: (?)
        - global_step
        - learning_rate_decay
        - change lr to tf.Variable
    """
    batch_size = 64
    n_epochs = 20
    lr = 0.005
    max_grad_norm = 5.
    clip_gradients = True
    encoder_hidden_units = 256
    decoder_hidden_units = 256
    attn_hidden_units = 128

class SequencePredictor(Model):

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input and target tensors
        """
        self.encoder_inputs = tf.placeholder(tf.int32,  shape = (None, self.config.max_length_x),
                                                        name  = "encoder_inputs")
        self.decoder_targets = tf.placeholder(tf.int32, shape =(None, self.config.max_length_y),
                                                        name  = "decoder_targets")
        self.decoder_inputs  = tf.placeholder(tf.int32, shape=(None, self.config.max_length_y),
                                                        name = "decoder_inputs")

        self.length_encoder_inputs = tf.placeholder(tf.int32, shape = (None),
                                                              name = "length_encoder_inputs")
        self.length_decoder_inputs = tf.placeholder(tf.int32, shape = (None),
                                                              name = "length_decoder_inputs")
        self.mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.config.max_length_y),
                                                        name = "mask_placeholder")

    def create_feed_dict(self, inputs_batch, length_encoder_batch, mask_batch = None,
                                                                   length_decoder_batch = None,
                                                                   decoder_inputs_batch = None,
                                                                   targets_batch = None):
        """
        Creates the feed_dict for the model.
        """

        if targets_batch is not None:
            feed_dict = {
                            self.encoder_inputs: inputs_batch,
                            self.decoder_inputs: decoder_inputs_batch,
                            self.decoder_targets: targets_batch,
                            self.length_encoder_inputs: length_encoder_batch,
                            self.length_decoder_inputs: length_decoder_batch,
                            self.mask_placeholder : mask_batch
                        }
        else:
            feed_dict = {
                            self.encoder_inputs: inputs_batch,
                            self.length_encoder_inputs: length_encoder_batch,
                            self.mask_placeholder : mask_batch
                        }

        return feed_dict

    def add_embeddings(self):
        """
        Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates a tf.Variable and initializes it with self.pretrained_embeddings.
            - Uses the encoder_inputs and decoder_inputs to index into the embeddings tensor,
              resulting in two tensor of shape (None, embedding_size).

        Returns:
            encoder_inputs_embedded: tf.Tensor of shape (None, embed_size)
            decoder_inputs_embedded: tf.Tensor of shape (None, embed_size)
        """
        E = tf.get_variable("E", initializer = self.pretrained_embeddings)
        encoder_inputs_embedded = tf.nn.embedding_lookup(E, self.encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(E, self.decoder_inputs)

        return encoder_inputs_embedded, decoder_inputs_embedded

    def add_prediction_op(self):
        """Runs a seq2seq model on the input using TensorFlows
        and returns the final state of the decoder as a prediction.

        Returns:
            train_preds: tf.Tensor of shape #TODO
            pred_outputs: tf.Tensor of shape #TODO
        """

        # Get embeddings
        encoder_inputs_embedded, decoder_inputs_embedded = self.add_embeddings()

        # Encoder (Bi-directional RNN)
        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.encoder_hidden_units)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.encoder_hidden_units)

        bi_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                                           backward_cell,
                                                                           encoder_inputs_embedded,
                                                                           dtype = tf.float32,
                                                                           sequence_length = self.length_encoder_inputs)

        encoder_outputs = tf.concat(bi_outputs, -1)

        # Helpers for train and inference
        self.length_decoder_inputs.set_shape([None])
        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded,
                                                         self.length_decoder_inputs)

        start_tokens = tf.fill([tf.shape(encoder_inputs_embedded)[0]], self.config.voc[START_TOKEN])
        end_token = self.config.voc[END_TOKEN]
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(E, start_tokens, end_token)

        # Decoder
        def decode(helper, scope, reuse = None):
            with tf.variable_scope(scope, reuse=reuse):
                self.length_encoder_inputs.set_shape([self.config.batch_size])
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
						self.config.encoder_hidden_units,
						memory = encoder_outputs,#)#,
	# To make sure attention weights are properly normalized (over non-padding positions only)
						memory_sequence_length = self.length_encoder_inputs)
                cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.config.decoder_hidden_units)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell,
                                            attention_mechanism,
                                            attention_layer_size = self.config.attn_hidden_units)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell,
                                                               self.config.voc_size,
                                                               reuse = reuse)
		#decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.config.decoder_hidden_units)
                #projection_layer = layers_core.Dense(self.config.voc_size, use_bias = False)
                initial_state = out_cell.zero_state(dtype = tf.float32,
                                                    batch_size = self.config.batch_size)
                maximum_iterations = self.config.max_length_y
                decoder = tf.contrib.seq2seq.BasicDecoder(out_cell,
                                                          helper,
                                                          initial_state = initial_state)
                                                          #output_layer = projection_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                            output_time_major = False,
                                                            maximum_iterations = maximum_iterations,
                                                            impute_finished = True)
                return outputs[0]
                #return outputs.rnn_output

        train_outputs = decode(train_helper, 'decode')
        pred_outputs = decode(pred_helper, 'decode', reuse=True)

        # pred_outputs = tf.argmax(decode(pred_helper, 'decode'),2)

        return train_outputs, pred_outputs

    def add_loss_op(self, preds):
        """
        Adds ops to compute the stepwise cross-entropy loss function.
        Args:
            preds: A tensor of shape (batch_size, 1) containing the last
            state of the neural network.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = self.decoder_targets
        current_ts = tf.to_int32(tf.minimum(tf.shape(y)[1], tf.shape(preds)[1]))
        y = tf.slice(y, begin=[0, 0], size=[-1, current_ts])
        target_weights = tf.sequence_mask(lengths=self.length_decoder_inputs,
                                          maxlen=current_ts,
                                          dtype=preds.dtype)

        preds = tf.slice(preds, begin=[0,0,0], size=[-1, current_ts, -1])

        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                   logits = preds)
        # target_weights = tf.sequence_mask(self.length_decoder_inputs,
                                          # self.config.max_length_y,
                                          # dtype = preds.dtype)
        loss = tf.reduce_mean(cross_ent*target_weights)
        # loss = tf.reduce_sum(cross_ent*target_weights)/tf.to_float(self.config.batch_size)
        # mask = tf.slice(self.mask_placeholder, begin = [0,0], size = [-1, current_ts])
        # loss2 = tf.reduce_sum(tf.boolean_mask(cross_ent, mask))/tf.to_float(self.config.batch_size)

        return loss

    def add_accuracy_op(self, preds):
        y = self.decoder_targets
        current_ts = tf.to_int32(tf.minimum(tf.shape(y)[1], tf.shape(preds)[1]))
        y = tf.slice(y, begin=[0, 0], size=[-1, current_ts])
        preds = tf.slice(preds, begin=[0,0,0], size=[-1, current_ts, -1])
        pred_idx = tf.to_int32(tf.argmax(preds, 2))		# [-1, self.out_seq_len]
        target_weights = tf.sequence_mask(lengths=self.length_decoder_inputs,
                                          maxlen=current_ts,
                                          dtype=preds.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_idx, y), tf.float32)*target_weights, name='acc')
        return acc

    def add_training_op(self, loss):
        """
        Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """

        optimizer = tf.train.AdamOptimizer(learning_rate = self.config.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        if self.config.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        self.grad_norm = tf.global_norm(gradients)
        train_op = optimizer.apply_gradients(zip(gradients,variables))

        return train_op

    def add_summary_op(self, loss, accuracy, dev = False):
        if not dev:
            loss_summary = tf.summary.scalar("train_loss", loss)
            accuracy_summary = tf.summary.scalar("train_accuracy", accuracy)
        else:
            loss_summary = tf.summary.scalar("dev_loss", loss)
            accuracy_summary = tf.summary.scalar("dev_accuracy", accuracy)
        return loss_summary, accuracy_summary

    def train_on_batch(self, sess, inputs_batch, targets_batch):
        """
        Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        inputs_batch_padded, _ = padded_batch_lr(inputs_batch,
                                                 self.config.max_length_x,
                                                 self.config.voc)
        length_inputs_batch = np.asarray([min(self.config.max_length_x,len(item))\
                                              for item in inputs_batch])

        if targets_batch is None:
            feed = self.create_feed_dict(inputs_batch_padded, length_inputs_batch)
        else:
            decoder_batch_padded, _ = padded_batch_lr(targets_batch,
                                                     self.config.max_length_y,
                                                     self.config.voc,
                                                     option = 'decoder_inputs')
            targets_batch_padded, mask_batch = padded_batch_lr(targets_batch,
                                                               self.config.max_length_y,
                                                               self.config.voc,
                                                               option = 'decoder_targets')
            length_decoder_batch = np.asarray([min(self.config.max_length_y, len(item)+1)\
                                                   for item in targets_batch])
            feed = self.create_feed_dict(inputs_batch_padded,
                                         length_inputs_batch,
                                         mask_batch,
                                         length_decoder_batch,
                                         decoder_batch_padded,
                                         targets_batch_padded)

        _, preds, loss, acc, loss_summ, acc_summ = sess.run([self.train_op,
                                                       self.train_pred,
                                                       self.loss,
                                                       self.accuracy,
                                                       self.loss_summary,
                                                       self.acc_summary], feed_dict=feed)
        preds = np.argmax(preds, 2)
        return preds, loss, acc, loss_summ, acc_summ

    def predict_on_batch(self, sess, inputs_batch, targets_batch = None):
        """
        Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, #TODO)
        Returns:
        e   predictions: np.ndarray of shape (n_samples, max_length_y)
        """
        inputs_batch_padded, _ = padded_batch_lr(inputs_batch,
                                                 self.config.max_length_x,
                                                 self.config.voc)
        length_inputs_batch = np.asarray([min(self.config.max_length_x,len(item))\
                                              for item in inputs_batch])
        if targets_batch is None:
            feed = self.create_feed_dict(inputs_batch_padded, length_inputs_batch)
        else:
            decoder_batch_padded, _ = padded_batch_lr(targets_batch,
                                                      self.config.max_length_y,
                                                      self.config.voc,
                                                      option = 'decoder_inputs')
            targets_batch_padded, mask_batch = padded_batch_lr(targets_batch,
                                                               self.config.max_length_y,
                                                               self.config.voc,
                                                               option = 'decoder_targets')

            length_decoder_batch = np.asarray([min(self.config.max_length_y, len(item)+1)\
                                                   for item in targets_batch])
            feed = self.create_feed_dict(inputs_batch_padded,
                                         length_inputs_batch,
                                         mask_batch,
                                         length_decoder_batch,
                                         decoder_batch_padded,
                                         targets_batch_padded)

        preds, dev_loss, dev_acc, dev_loss_summ, dev_acc_summ = sess.run([self.infer_pred,
                                                                          self.dev_loss,
                                                                          self.dev_accuracy,
                                                                          self.dev_loss_summary,
                                                                          self.dev_acc_summary],
                                                                          feed_dict=feed)
        preds = np.argmax(preds,2)
        return preds, dev_loss, dev_acc, dev_loss_summ, dev_acc_summ

    def run_epoch(self, sess, saver, train, dev):
        prog = Progbar(target= int(len(train) / self.config.batch_size))
        train_preds, losses, accs, refs = [], [], [], []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            _, targets_batch = batch
            train_pred, loss, acc, loss_summ, acc_summ = self.train_on_batch(sess, *batch)
            train_pred = list(train_pred)
            losses.append(loss)
            accs.append(acc)
            train_preds += train_pred
            refs += list(targets_batch)
            prog.update(i + 1, [("train loss", loss), ("train acc", acc)])

        train_preds = [tokens_to_sentences(pred, self.config.idx2word) for pred in train_preds]
        refs  = [tokens_to_sentences(ref, self.config.idx2word) for ref in refs]

        train_f1, _, _ = rouge_n(train_preds, refs)
        print("- train rouge f1: {}".format(train_f1))

        print("\nEvaluating on dev set...")
        dev_preds, refs, dev_losses, dev_accs = [], [], [], []
        prog_dev = Progbar(target= int(len(dev) / self.config.batch_size))
        for i, batch in enumerate(minibatches(dev, self.config.batch_size)):
            _, targets_batch = batch
            dev_pred, dev_loss, dev_acc, dev_loss_summ, dev_acc_summ = self.predict_on_batch(sess, *batch)
            dev_pred = list(dev_pred)
            dev_losses.append(dev_loss)
            dev_accs.append(dev_acc)
            dev_preds += dev_pred
            refs += list(targets_batch)
            prog_dev.update(i + 1, [("dev loss", dev_loss), ("dev_acc", dev_acc)])

        dev_preds = [tokens_to_sentences(pred, self.config.idx2word) for pred in dev_preds]
        refs  = [tokens_to_sentences(ref, self.config.idx2word) for ref in refs]

        dev_f1, _, _ = rouge_n(dev_preds, refs)
        print("- dev rouge f1: {}".format(dev_f1))
        return losses, accs, dev_losses, dev_accs, loss_summ, acc_summ, dev_loss_summ, dev_acc_summ, dev_f1

    def fit(self, sess, saver, train, dev):
        losses, grad_norms, predictions, dev_losses = [], [], [], []
        accs, dev_accs = [], []
        best_dev_ROUGE = -1.0
        # best_dev_loss = np.inf
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, acc, dev_loss, dev_acc, lsumm, asumm, d_lsumm, d_asumm, f1 = self.run_epoch(sess, saver, train, dev)
            if writer:
                print("Saving graph in ./data/graph/summaries")
                writer.add_summary(lsumm, global_step = epoch)
                writer.add_summary(asumm, global_step = epoch)
                writer.add_summary(d_lsumm, global_step = epoch)
                writer.add_summary(d_asumm, global_step = epoch)

            #dev_loss_epoch = np.mean(np.asarray(dev_loss))
            #if dev_loss_epoch < best_dev_loss:
            if f1 > best_dev_ROUGE:
                best_dev_ROUGE = f1
                #best_dev_loss = dev_loss_epoch
                if saver:
                    print("New best dev ROUGE! Saving model in ./data/weights/model.weights")
                    #print("New best dev loss! Saving model in ./data/weights/model.weights")
                    saver.save(sess, './data/weights/model.weights')

            losses.append(loss)
            dev_losses.append(dev_loss)
            accs.append(acc)
            dev_accs.append(dev_acc)

        return losses, dev_losses, accs, dev_accs

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_targets = None
        self.length_encoder_inputs = None
        self.length_decoder_inputs = None
        self.grad_norm = None
        self.mask_placeholder = None
        self.build()

def make_losses_plot(train_loss, dev_loss, fname, title, ylab):
    plt.clf()
    plt.title(title)

    plt.plot(np.arange(len(train_loss)), train_loss, color = 'coral', label="train")
    plt.plot(np.arange(len(dev_loss)), dev_loss, color = 'mediumvioletred', label="dev")
    plt.ylabel(ylab)
    plt.xlabel('iteration')
    plt.legend()
    output_path = "{}.png".format(fname)
    plt.savefig(output_path)

if __name__ == '__main__':

    # Get data and embeddings
    start = time.time()
    print("Loading data...")
    if debug:
        train, dev, test, E, voc, _, _ = load_data('train_filtered_small.txt',
                                                   'dev_filtered_small.txt',
                                                   'test_filtered_small.txt')
    else:
        train, dev, test, E, voc, _, _ = load_data('train_filtered.txt',
                                                   'dev_filtered.txt',
                                                   'test_filtered.txt')

    print("Took {} seconds to load data".format(time.time() - start))

    # Set up some parameters.
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()
    config.voc_size = len(voc)
    config.embedding_size = E.shape[1]
    config.max_length_x = 200
    config.max_length_y = 6
    config.voc = voc
    config.idx2word = dict([[v,k] for k,v in voc.items()])

    # Fix length for attention to work
    m = len(train)%config.batch_size
    train = train[:-(m)] if m > 0 else train
    m = len(dev)%config.batch_size
    dev = dev[:-(m)] if m > 0 else dev
    m = len(test)%config.batch_size
    test = test[:-(m)] if m > 0 else test

    UNK_IDX = voc[UNK_TOKEN]
    START_IDX = voc[START_TOKEN]
    END_IDX = voc[END_TOKEN]
    PAD_IDX = voc[PAD_TOKEN]

    # Create directory for saver
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')
    if not os.path.exists('./data/graphs/'):
        os.makedirs('./data/graphs/')
    if not os.path.exists('./data/predictions/'):
        os.makedirs('./data/predictions/')

    # Build model
    with tf.Graph().as_default() as graph:
        start = time.time()
        print("Building model...")
        model = SequencePredictor(config, E)
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('./data/graphs', tf.get_default_graph())
        print("Took {} seconds to build model".format(time.time() - start))

    graph.finalize()

    with tf.Session(graph = graph) as sess:
        sess.run(init_g)
        sess.run(init_l)
        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        losses, dev_losses, accs, dev_accs = model.fit(sess,
                                                       saver,
                                                       train,
                                                       dev)
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        saver.restore(sess, './data/weights/model.weights')
        print("Final evaluation on test set")
        preds = []
        refs = []
        test_losses = []
        test_accs = []
        for batch in minibatches(test, model.config.batch_size):
            inputs_batch, targets_batch = batch
            pred, test_loss, test_acc, _, _ = model.predict_on_batch(sess, *batch)
            pred = list(pred)
            preds += pred
            refs += list(targets_batch)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        mean_test_loss = np.mean(np.asarray(test_losses))
        preds = [tokens_to_sentences(pred, model.config.idx2word) for pred in preds]
        refs  = [tokens_to_sentences(ref, model.config.idx2word) for ref in refs]

        f1, _, _ = rouge_n(preds, refs)
        print("- test ROUGE: {}".format(f1))
        print("- test loss: {}".format(mean_test_loss))
        print("Writing predictions")
        fname = './data/predictions' + str(date.today()) + '.txt'
        with open(fname, 'w') as f:
            for pred, ref in zip(preds, refs):
                f.write(pred + '\t' + ref)
                f.write('\n')
        print("Done!")

    plot_fname = 'loss_plot-' + str(date.today())
    plosses = [np.mean(np.array(item)) for item in losses]
    pdev_losses = [np.mean(np.array(item)) for item in dev_losses]

    print("Writing losses to file ...")
    fname = 'losses-' + str(date.today()) + '.txt'
    with open(fname, 'w') as f:
        for tr_loss, dev_loss in zip(plosses, pdev_losses):
            f.write(str(tr_loss) + '\t' + str(dev_loss))
            f.write('\n')

    make_losses_plot(plosses, pdev_losses, plot_fname, 'Train vs dev loss', 'loss')

    plot_fname = 'acc_plot-' + str(date.today())
    paccs = [np.mean(np.array(item)) for item in accs]
    pdev_accs = [np.mean(np.array(item)) for item in dev_accs]
    make_losses_plot(paccs, pdev_accs, plot_fname, 'Train vs dev accuracy', 'acc')

    print("Writing accuracies to file ...")
    fname = 'accs-' + str(date.today()) + '.txt'
    with open(fname, 'w') as f:
        for tr_acc, dev_acc in zip(paccs, pdev_accs):
            f.write(str(tr_acc) + '\t' + str(dev_acc))
            f.write('\n')
    writer.close()

