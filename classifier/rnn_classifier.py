from __future__ import print_function

import tensorflow as tf

'''
To classify text using a BiDirectional LSTM
'''
# Parameters
learning_rate = 0.01
training_iters = 1000000
batch_size = 16
display_step = 15000 / batch_size
n_layers = 1
n_hidden = 30  # hidden layer num of features
embedding_size = 64
dropout = 0.5


class BiRNNClassifier:
    def __init__(self, sequence_length, num_classes, vocab_size):
        self.n_steps = sequence_length
        self.n_classes = num_classes
        self.vocab_size = vocab_size
        # Define weights
        self.weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2 * n_hidden, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_pred_op()
        self.add_cost_op()
        self.add_optimize_op()
        self.add_accuracy_op()

    def add_placeholders(self):
        # tf Graph input
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, self.n_steps])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])

    def add_pred_op(self):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)


        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Define n_layers of lstm cells
        m_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(m_lstm_cell, m_lstm_cell, self.word_embeddings,
                                                                    sequence_length=self.sequence_lengths,
                                                                    dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.nn.dropout(output, self.dropout)

        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Softmax, using rnn inner loop last output
        self.pred = tf.nn.softmax(tf.matmul(last, self.weights['out']) + self.biases['out'])

    def add_cost_op(self):
        # Define loss and optimizer
        cross_entropy = -tf.reduce_sum(self.y * tf.log(self.pred))
        self.cost = cross_entropy

    def add_optimize_op(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def add_accuracy_op(self):
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            W = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0), name="W")
            word_embeddings = tf.nn.embedding_lookup(W, self.x, name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    @staticmethod
    def get_sequence_lengths(x):
        return [len(word_ids) for word_ids in x]

    def train(self, dataset):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = dataset.train.next_batch(batch_size)
                seq_lengths = BiRNNClassifier.get_sequence_lengths(batch_x)
                # Run optimization op
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: dropout,
                                                    self.sequence_lengths: seq_lengths})
                if step % display_step == 0:
                    # Calculate dev accuracy
                    dev_x, dev_y = dataset.dev.all()
                    seq_lengths = BiRNNClassifier.get_sequence_lengths(dev_x)

                    acc = sess.run(self.accuracy, feed_dict={self.x: dev_x, self.y: dev_y, self.dropout: 1.0,
                                                             self.sequence_lengths: seq_lengths})
                    #  Calculate batch loss
                    loss = sess.run(self.cost, feed_dict={self.x: dev_x, self.y: dev_y, self.dropout: 1.0,
                                                          self.sequence_lengths: seq_lengths})
                    print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                        loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

            # Calculate accuracy for dev data
            # Fixme read test dataset
            test_x, test_y = dataset.dev.all()
            test_x = test_x.reshape((len(test_x), self.n_steps, 1))
            print("Testing Accuracy:", sess.run(self.accuracy, feed_dict={self.x: test_x, self.y: test_y}))
