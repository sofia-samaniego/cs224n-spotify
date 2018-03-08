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

import matplotlib.pyplot as plt
import logging
from datetime import datetime, date

logger = logging.getLogger("project.milestone")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

global UNK_IDX, START_IDX, END_IDX, PAD_IDX
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
    n_epochs = 30
    lr = 0.005
    max_grad_norm = 5.
    clip_gradients = True
    encoder_hidden_units = 256
    decoder_hidden_units = 256

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

        # Encoder
        encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.config.encoder_hidden_units)

        encoder_inputs_embedded, decoder_inputs_embedded = self.add_embeddings()
        initial_state = encoder_cell.zero_state(tf.shape(encoder_inputs_embedded)[0],
                                                dtype = tf.float32)
        _, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                   encoder_inputs_embedded,
        #                                           initial_state = initial_state,
                                                   sequence_length = self.length_encoder_inputs,
                                                   dtype = tf.float32)

        # Helpers for train and inference
        self.length_decoder_inputs.set_shape([None])
        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded,
                                                         self.length_decoder_inputs)

        start_tokens = tf.fill([tf.shape(encoder_inputs_embedded)[0]], self.config.voc[START_TOKEN])
        end_token = self.config.voc[END_TOKEN]
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(E, start_tokens, end_token)

        # Decoder
        def decode(helper, scope, reuse = None):
            # Here could add attn_cell, etc. (see https://gist.github.com/ilblackdragon/)
            decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.config.decoder_hidden_units)
            projection_layer = layers_core.Dense(self.config.voc_size, use_bias = False)
            maximum_iterations = self.config.max_length_y
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      encoder_final_state,
                                                      output_layer = projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                maximum_iterations = maximum_iterations,
                                                impute_finished = True)
            return outputs.rnn_output

        train_outputs = decode(train_helper, 'decode')
        pred_outputs = decode(pred_helper, 'decode')

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
        loss = tf.reduce_sum(cross_ent*target_weights)/tf.to_float(self.config.batch_size)
        mask = tf.slice(self.mask_placeholder, begin = [0,0], size = [-1, current_ts])
        loss2 = tf.reduce_sum(tf.boolean_mask(cross_ent, mask))/tf.to_float(self.config.batch_size)

        return loss

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

    def add_summary_op(self, loss):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            summary_op = tf.summary.merge_all()
        return summary_op

    def train_on_batch(self, sess, inputs_batch, targets_batch):
        """
        Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        inputs_batch_padded, _ = padded_batch_lr(inputs_batch,
                                                 self.config.max_length_x,
                                                 self.config.voc)
        length_inputs_batch = np.asarray([min(self.config.max_length_x,len(item)) for item in inputs_batch])

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
            length_decoder_batch = np.asarray([min(self.config.max_length_y, len(item)+1) for item in targets_batch])
            feed = self.create_feed_dict(inputs_batch_padded,
                                         length_inputs_batch,
                                         mask_batch,
                                         length_decoder_batch,
                                         decoder_batch_padded,
                                         targets_batch_padded)

        _, loss, grad_norm, summ = sess.run([self.train_op, self.loss, self.grad_norm, self.summary_op], feed_dict=feed)
        preds = np.asarray(sess.run([self.train_pred], feed_dict = feed))
        preds = np.argmax(preds[0],2)
        print("\n")
        print(tokens_to_sentences(targets_batch[0], self.config.idx2word))
        print(tokens_to_sentences(preds[0], self.config.idx2word))
        return loss, grad_norm, summ

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
        length_inputs_batch = np.asarray([min(self.config.max_length_x,len(item)) for item in inputs_batch])
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

            #length_decoder_batch = np.asarray([min(self.config.max_length_y, len(item)) for item in targets_batch])
            length_decoder_batch = np.asarray([self.config.max_length_y for item in targets_batch])
            feed = self.create_feed_dict(inputs_batch_padded,
                                         length_inputs_batch,
                                         mask_batch,
                                         length_decoder_batch,
                                         decoder_batch_padded,
                                         targets_batch_padded)
        predictions, dev_loss = sess.run([self.infer_pred, self.dev_loss], feed_dict=feed)
        preds = np.argmax(predictions,2)
        print(tokens_to_sentences(targets_batch[0], self.config.idx2word))
        print(tokens_to_sentences(preds[0], self.config.idx2word))
        return preds, dev_loss

    def run_epoch(self, sess, saver, train, dev):
        prog = Progbar(target= int(len(train) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            loss, grad_norm, summ = self.train_on_batch(sess, *batch)
            losses.append(loss)
            grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss)])

        print("\nEvaluating on dev set...")
        predictions = []
        references = []
        dev_losses = []
        prog_dev = Progbar(target= int(len(dev) / self.config.batch_size))
        for i, batch in enumerate(minibatches(dev, self.config.batch_size)):
            inputs_batch, targets_batch = batch
            # prediction = list(self.predict_on_batch(sess, inputs_batch))
            prediction, dev_loss = self.predict_on_batch(sess, *batch)
            prediction = list(prediction)
            dev_losses.append(dev_loss)
            predictions += prediction
            references += list(targets_batch)
            prog_dev.update(i + 1, [("dev loss", dev_loss)])

        predictions = [tokens_to_sentences(pred, self.config.idx2word) for pred in predictions]
        references  = [tokens_to_sentences(ref, self.config.idx2word) for ref in references]

        f1, _, _ = rouge_n(predictions, references)
        print("- dev rouge f1: {}".format(f1))
        return losses, grad_norms, summ, predictions, f1, dev_losses

    def fit(self, sess, saver, train, dev):
        losses, grad_norms, predictions, dev_losses = [], [], [], []
        # best_dev_ROUGE = -1.0
        best_dev_loss = np.inf
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, grad_norm, summ, preds, f1, dev_loss = self.run_epoch(sess,
                                                                        saver,
                                                                        train,
                                                                        dev)
            if writer:
                print("Saving graph in ./data/graph/loss.summary")
                writer.add_summary(summ, global_step = epoch)
            # if f1 > best_dev_ROUGE:
            if dev_loss[-1] < best_dev_loss:
                best_dev_loss = dev_loss[-1]
                # best_dev_ROUGE = f1
                if saver:
                    print("New best dev loss! Saving model in ./data/weights/model.weights")
                    saver.save(sess, './data/weights/model.weights')
            losses.append(loss)
            dev_losses.append(dev_loss)
            grad_norms.append(grad_norm)
            predictions.append(preds)
        return losses, grad_norms, predictions, dev_losses

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

def make_losses_plot(train_loss, dev_loss, fname):
    plt.clf()
    plt.title('Training vs. Dev Loss')

    plt.plot(np.arange(len(train_loss)), train_loss, color = 'coral', label="train")
    plt.plot(np.arange(len(dev_loss)), dev_loss, color = 'mediumvioletred', label="dev")
    plt.ylabel("iteration")
    plt.xlabel("loss")
    plt.legend()
    output_path = "{}.png".format(fname)
    plt.savefig(output_path)

if __name__ == '__main__':

    # Get data and embeddings
    start = time.time()
    print("Loading data...")
    if debug:
        train, dev, test, E, voc, max_x, max_y = load_data('train_all_small.txt',
                                                           'dev_all_small.txt',
                                                           'test_all.txt',
                                                           'max_lengths_all.txt')
    else:
        train, dev, test, E, voc, max_x, max_y = load_data('train_all.txt',
                                                           'dev_all.txt',
                                                           'test_all.txt',
                                                           'max_lengths_all.txt')

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
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('./data/graphs', tf.get_default_graph())
        print("Took {} seconds to build model".format(time.time() - start))

    graph.finalize()

    with tf.Session(graph = graph) as sess:
        sess.run(init)
        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        losses, grad_norms, predictions, dev_losses = model.fit(sess,
                                                                saver,
                                                                train,
                                                                dev)
        if not debug:
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(sess, './data/weights/model.weights')
            print("Final evaluation on test set")
            predictions = []
            references = []
            test_losses = []
            for batch in minibatches(test, model.config.batch_size):
                inputs_batch, targets_batch = batch
                #prediction = list(model.predict_on_batch(sess, inputs_batch))
                prediction, test_loss = model.predict_on_batch(sess, *batch)
                prediction = list(prediction)
                predictions += prediction
                references += list(targets_batch)
                test_losses.append(test_loss)

            predictions = [tokens_to_sentences(pred, model.config.idx2word) for pred in predictions]
            references  = [tokens_to_sentences(ref, model.config.idx2word) for ref in references]

            f1, _, _ = rouge_n(predictions, references)
            print("- test ROUGE: {}".format(f1))
            print("- test loss: {}".format(test_losses[-1]))
            print("Writing predictions")
            fname = 'predictions' + str(date.today()) + '.txt'
            with open(fname, 'w') as f:
                for pred, ref in zip(predictions, references):
                    f.write(pred + '\t' + ref)
                    f.write('\n')
            print("Done!")

    plot_fname = 'loss' + str(date.today())
    plosses = [np.mean(np.array(item)) for item in losses]
    pdev_losses = [np.mean(np.array(item)) for item in dev_losses]
    make_losses_plot(plosses, pdev_losses, plot_fname)

    writer.close()

