"""
This script converts the tracklists of the playlists into sequences
of word indices into the embedding matrix from the 100-dimensional
pre-trained GloVe vectors found here:
    http://nlp.stanford.edu/projects/glove/
"""

import re
import sys
import tensorflow as tf
import numpy as np
import time
import os

from model import Model
from data_util import load_and_preprocess_data, UNK_TOKEN, START_TOKEN, END_TOKEN, PAD_TOKEN
from util import Progbar, minibatches
from rouge import rouge_n

import logging
from datetime import datetime

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
    batch_size = 100
    n_epochs = 2
    lr = 0.2
    max_grad_norm = 5.
    cell = 'lstm'
    clip_gradients = False
    encoder_hidden_units = 20
    decoder_hidden_units = 20

def padded_batch(batch, max_length, voc = None, option = None):
    padded_batch = []
    for item in batch:
        if len(item) >= max_length:
            padded_item = item[:max_length]
        else:
            padded_item = item + [voc[PAD_TOKEN]]*(max_length - len(item))
        if option == 'decoder_inputs':
            padded_item = padded_item[0:-1]
            padded_item.insert(0, voc[START_TOKEN])
        if option == 'decoder_targets':
            padded_item = padded_item[0:-1]
            padded_item += [voc[END_TOKEN]]
        padded_batch.append(padded_item)
    return padded_batch


class SequencePredictor(Model):

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input and target tensors
        """
        self.encoder_inputs = tf.placeholder(tf.int32,  shape = (None, self.config.max_length_x),
                                                          name  = "encoder_inputs")
        self.decoder_targets = tf.placeholder(tf.int32, shape =(None, self.config.max_length_y),
                                                          name  = "decoder_targets")
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, self.config.max_length_y),
                                                         name = "decoder_inputs")

    def create_feed_dict(self, inputs_batch, targets_batch = None):
        """
        Creates the feed_dict for the model.
        """
        encoder_inputs_padded = padded_batch(inputs_batch,
                                             self.config.max_length_x,
                                             self.config.voc)

        if targets_batch is not None:
            decoder_inputs_padded = padded_batch(targets_batch,
                                                 self.config.max_length_y,
                                                 self.config.voc,
                                                 option = 'decoder_inputs')
            decoder_targets_padded = padded_batch(targets_batch,
                                                  self.config.max_length_y,
                                                  self.config.voc,
                                                  option = 'decoder_targets')
            feed_dict = {
                            self.encoder_inputs: encoder_inputs_padded,
                            self.decoder_inputs: decoder_inputs_padded,
                            self.decoder_targets: decoder_targets_padded
                        }
        else:
            feed_dict = {
                            self.encoder_inputs: encoder_inputs_padded
                        }

        return feed_dict

    def add_embeddings(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
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
        """Runs an rnn on the input using TensorFlows's
        @tf.nn.dynamic_rnn function, and returns the final state as a prediction.

        TODO:
            - Call tf.nn.dynamic_rnn using @cell below. See:
              https://www.tensorflow.org/api_docs/python/nn/recurrent_neural_networks
            - Apply a sigmoid transformation on the final state to
              normalize the inputs between 0 and 1.

        Returns:
            preds: tf.Tensor of shape (batch_size, 1)
        """

        # Encoder
        encoder_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_units)
        encoder_inputs_embedded, decoder_inputs_embedded = self.add_embeddings()
        _, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype = tf.float32)

        # Decoder
        decoder_cell = tf.contrib.rnn.LSTMCell(self.config.decoder_hidden_units)
        decoder_outputs, decoder_state = tf.nn.dynamic_rnn(decoder_cell,
                                                           decoder_inputs_embedded,
                                                           initial_state = encoder_final_state,
                                                           dtype = tf.float32,
                                                           scope = "plain_decoder")

        preds = tf.contrib.layers.linear(decoder_outputs, self.config.voc_size)
        #decoder_prediction = tf.argmax(preds, 2)

        return preds

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
        stepwise_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                                logits = preds)
        loss = tf.reduce_mean(stepwise_cross_entropy)

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """

        optimizer = tf.train.AdamOptimizer(learning_rate = self.config.lr)
        # train_op = optimizer.minimize(loss)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        if self.config.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        self.grad_norm = tf.global_norm(gradients)
        train_op = optimizer.apply_gradients(zip(gradients,variables))

        return train_op

    def train_on_batch(self, sess, inputs_batch, targets_batch):
        """Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        feed = self.create_feed_dict(inputs_batch, targets_batch=targets_batch)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def run_epoch(self, sess, train):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            loss, grad_norm = self.train_on_batch(sess, *batch)
            losses.append(loss)
            grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss)])

        return losses, grad_norms

    def fit(self, sess, train):
    #def fit(self, sess, saver, train, dev):
        losses, grad_norms = [], []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, grad_norm = self.run_epoch(sess, train)
            losses.append(loss)
            grad_norms.append(grad_norm)

        return losses, grad_norms

    # def fit(self, sess, saver, parser, train_examples, dev_set):
        # best_dev_UAS = 0
        # for epoch in range(self.config.n_epochs):
            # print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            # dev_UAS = self.run_epoch(sess, parser, train_examples, dev_set)
            # if dev_UAS > best_dev_UAS:
                # best_dev_UAS = dev_UAS
                # if saver:
                    # print "New best dev UAS! Saving model in ./data/weights/parser.weights"
                    # saver.save(sess, './data/weights/parser.weights')
            # print

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_targets = None
        self.grad_norm = None
        self.build()

if __name__ == '__main__':

    # Get data and embeddings
    start = time.time()
    print("Loading data...")
    train, dev, test, _, _, _, max_x, max_y, E, voc = load_and_preprocess_data(output = 'tokens_debug.txt', debug = True)
    print("Took {} seconds to load data".format(time.time() - start))

    # Set up some parameters.
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()
    config.voc_size = len(voc)
    config.embedding_size = E.shape[1]
    config.max_length_x = 250
    config.max_length_y = 11
    config.voc = voc

    UNK_IDX = voc[UNK_TOKEN]
    START_IDX = voc[START_TOKEN]
    END_IDX = voc[END_TOKEN]
    PAD_IDX = voc[PAD_TOKEN]

    # Create directory for saver
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    # Build model
    with tf.Graph().as_default() as graph:
        start = time.time()
        print("Building model...")
        model = SequencePredictor(config, E)
        init = tf.global_variables_initializer()
        #saver = tf.train.Saver()
        print("Took {} seconds to build model".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph = graph) as sess:
        sess.run(init)
        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        losses, grad_norms = model.fit(sess, train)

# def main(debug=True):

    # with tf.Session(graph=graph) as session:
        # parser.session = session
        # session.run(init_op)

        # print 80 * "="
        # print "TRAINING"
        # print 80 * "="
        # model.fit(session, saver, parser, train_examples, dev_set)

        # if not debug:
            # print 80 * "="
            # print "TESTING"
            # print 80 * "="
            # print "Restoring the best model weights found on the dev set"
            # saver.restore(session, './data/weights/parser.weights')
            # print "Final evaluation on test set",
            # UAS, dependencies = parser.parse(test_set)
            # print "- test UAS: {:.2f}".format(UAS * 100.0)
            # print "Writing predictions"
            # with open('q2_test.predicted.pkl', 'w') as f:
                # cPickle.dump(dependencies, f, -1)
            # print "Done!"



