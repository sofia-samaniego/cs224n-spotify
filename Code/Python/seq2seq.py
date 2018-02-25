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

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    max_length_x = 1500 # Length of input sequence used.
    max_length_y = 10
    batch_size = 100
    n_epochs = 40
    lr = 0.2
    max_grad_norm = 5.
    cell = 'lstm'
    clip_gradients = False

class SequencePredictor(Model):
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                    self.config.max_length_x,
                                                                    1), name="x")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                    self.config.max_length_y,
                                                                    1), name="y")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates a tf.Variable and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/tf/reshape

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
        E = tf.get_variable("E", initializer = self.pretrained_embeddings) # each row is embedding vector for a particular word
        X = tf.nn.embedding_lookup(E, self.input_placeholder)
        embeddings = tf.reshape(X, [-1,self.config.n_features*self.config.embed_size])
        ### END YOUR CODE
        return embeddings

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

        # Pick out the cell to use here.
        if self.config.cell == "rnn":
            cell = RNNCell(1, 1)
        elif self.config.cell == "gru":
            cell = GRUCell(1, 1)
        elif self.config.cell == "lstm":
            cell = tf.nn.rnn_cell.LSTMCell(1)
        else:
            raise ValueError("Unsupported cell type.")

        x = self.add_embedding()
        ### YOUR CODE HERE (~2-3 lines)
        _, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
        preds = tf.nn.sigmoid(state)
        ### END YOUR CODE

        return preds #state # preds

    def add_loss_op(self, preds):
        """Adds ops to compute the loss function.
        Here, we will use a simple l2 loss.

        Tips:
            - You may find the functions tf.reduce_mean and tf.l2_loss
              useful.

        Args:
            pred: A tensor of shape (batch_size, 1) containing the last
            state of the neural network.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = self.labels_placeholder

        ### YOUR CODE HERE (~1-2 lines)
        loss = tf.reduce_mean(tf.nn.l2_loss(y - preds))
        ### END YOUR CODE

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        TODO:
            - Get the gradients for the loss from optimizer using
              optimizer.compute_gradients.
            - if self.clip_gradients is true, clip the global norm of
              the gradients using tf.clip_by_global_norm to self.config.max_grad_norm
            - Compute the resultant global norm of the gradients using
              tf.global_norm and save this global norm in self.grad_norm.
            - Finally, actually create the training operation by calling
              optimizer.apply_gradients.
			- Remember to clip gradients only if self.config.clip_gradients
			  is True.
			- Remember to set self.grad_norm
        See: https://www.tensorflow.org/api_docs/python/train/gradient_clipping
        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)

        ### YOUR CODE HERE (~6-10 lines)

        # - Remember to clip gradients only if self.config.clip_gradients
        # is True.
        # - Remember to set self.grad_norm

        # optimizer = tf.train.AdamOptimizer(learning_rate = self.config.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        if self.config.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        self.grad_norm = tf.global_norm(gradients)
        train_op = optimizer.apply_gradients(zip(gradients,variables))

        assert self.grad_norm is not None, "grad_norm was not set properly!"
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
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
        losses, grad_norms = [], []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, grad_norm = self.run_epoch(sess, train)
            losses.append(loss)
            grad_norms.append(grad_norm)

        return losses, grad_norms

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.grad_norm = None
        self.build()

def do_sequence_prediction():
    # Set up some parameters.
    config = Config()

    # Get data
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)

    with tf.Graph().as_default():
        # You can change this around, but make sure to reset it to 41 when
        # submitting.
        tf.set_random_seed(59)

        # Initializing RNNs weights to be very large to showcase
        # gradient clipping.


        start = time.time()
        model = SequencePredictor(config, embeddings)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            losses, grad_norms = model.fit(session, data)

if __name__ == '__main__':
    do_sequence_prediction()

# def main(debug=True):
    # print 80 * "="
    # print "INITIALIZING"
    # print 80 * "="
    # config = Config()
    # parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    # if not os.path.exists('./data/weights/'):
        # os.makedirs('./data/weights/')

    # with tf.Graph().as_default() as graph:
        # print "Building model...",
        # start = time.time()
        # model = SequencePredictor(config, embeddings)
        # parser.model = model
        # init_op = tf.global_variables_initializer()
        # saver = None if debug else tf.train.Saver()
        # print "took {:.2f} seconds\n".format(time.time() - start)
    # graph.finalize()

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



