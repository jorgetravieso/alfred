import tensorflow as tf
from tensorflow.contrib import rnn as contrib_rnn

char_dims = 300
char_hidden_size = 100

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, sequence_lengths, hidden_units, number_of_layers):
        n_hidden = hidden_units
        n_layers = number_of_layers

        def StackedLSTMCell():
            def LSTMCell():
                return tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                # cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                # return contrib_rnn.AttentionCellWrapper(cell, 7)
            return tf.contrib.rnn.MultiRNNCell([LSTMCell() for _ in range(n_layers)], state_is_tuple=True)

        # Define stacked forward and backward LSTM cells.
        # Forward direction cell
        with tf.name_scope("fw-" + scope), tf.variable_scope("fw-" + scope):
            print(tf.get_variable_scope().name)
            lstm_fw_cell_m = StackedLSTMCell()
        # Backward direction cell
        with tf.name_scope("bw-" + scope), tf.variable_scope("bw-" + scope):
            print(tf.get_variable_scope().name)
            lstm_bw_cell_m = StackedLSTMCell()
        # Get lstm cell output
        with tf.name_scope("bi-lstm-" + scope), tf.variable_scope("bi-lstm-" + scope):
            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, sequence_length=sequence_lengths, dtype=tf.float32)
            output = tf.concat([fw_output, bw_output], axis=-1)
            output = tf.nn.dropout(output, dropout)

        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        return last

    def RNN(self, x, dropout, scope, sequence_lengths, hidden_units, number_of_layers):
        n_hidden = hidden_units
        n_layers = number_of_layers

        def StackedLSTMCell():
            def LSTMCell():
                return tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                # cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                # return contrib_rnn.AttentionCellWrapper(cell, 7)

            return tf.contrib.rnn.MultiRNNCell([LSTMCell() for _ in range(n_layers)], state_is_tuple=True)

        # Define stacked forward LSTM cells
        # Forward direction cell
        with tf.name_scope("fw-" + scope), tf.variable_scope("fw-" + scope):
            print(tf.get_variable_scope().name)
            lstm_fw_cell_m = StackedLSTMCell()
        # Get lstm cell output
        with tf.name_scope("lstm-" + scope), tf.variable_scope("lstm-" + scope):
            output, _ = tf.nn.dynamic_rnn(lstm_fw_cell_m, x, sequence_length=sequence_lengths, dtype=tf.float32)
            output = tf.nn.dropout(output, dropout)
            # Select last output.
            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
        return last

    def softmax(self, h1, h2):
        v = tf.concat([h1, h2, tf.squared_difference(h1, h2), tf.multiply(h1, h2)], axis=-1)
        return tf.nn.softmax(tf.matmul(v, self.weights['out']) + self.biases['out'])

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        loss = tf.reduce_sum(tmp + tmp2) / batch_size / 2
        tf.summary.scalar('contrastive_loss', tf.reduce_mean(loss))
        return loss

    def mean_squared_error(self, y, pred):
        mse = tf.reduce_mean(tf.squared_difference(y[1], pred[1]))
        tf.summary.scalar('mse', tf.reduce_mean(mse))
        return mse

    def contrastive_loss_2(self, y, d, batch_size):
        loss = y * tf.square(tf.maximum(0., y - d)) + (1 - y) * d
        loss = 0.5 * tf.reduce_mean(loss)
        tf.summary.scalar('contrastive_loss_2', tf.reduce_mean(loss))
        return loss

    def contrastive_loss_3(self, y, d, batch_size, margin=1.0):
        one = tf.constant(1.0)
        between_class = tf.exp(tf.multiply(one - y, tf.square(d)))  # (1-Y)*(d^2)
        max_part = tf.square(tf.maximum(margin - d, 0))

        within_class = tf.multiply(y, max_part)                     # (Y) * max((margin - d)^2, 0)

        loss = 0.5 * tf.reduce_mean(within_class + between_class)
        tf.summary.scalar('contrastive_loss_3', tf.reduce_mean(loss))
        tf.summary.scalar('contrastive_loss_within_class', tf.reduce_mean(within_class))
        tf.summary.scalar('contrastive_loss_between_class', tf.reduce_mean(between_class))
        return loss

    def cross_entropy_loss(self, y, pred):
        cross_entropy = -tf.reduce_sum(y * tf.log(pred))
        tf.summary.scalar('cross_entropy', tf.reduce_mean(cross_entropy))
        return cross_entropy

    def __init__(
            self, vocab_size, embedding_size, hidden_units, number_of_layers, l2_reg_lambda, batch_size, word_embeddings, n_chars, max_length, use_bi_lstm=True, embedding_type="static", loss_function="cross_entropy"):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, max_length], name="input_x1")
        self.input_x1_lens = tf.placeholder(tf.int32, [None], name="input_x1_lens")
        self.input_x2 = tf.placeholder(tf.int32, [None, max_length], name="input_x2")
        self.input_x2_lens = tf.placeholder(tf.int32, [None], name="input_x2_lens")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # shape = (batch size, max length of sentence, max length of word)
        #TODO self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")
        self.batch_size = batch_size



        # Define weights
        self.weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            # Todo are we really using the bidirectional output? If yes this should be 4X (2Xfw + 2Xbw)
            'out': tf.Variable(tf.random_normal([4 * (2 if use_bi_lstm else 1) * hidden_units, 2]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([2]))
        }

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")


        # Embedding layer
        with tf.name_scope("embedding"):
            if embedding_type in 'static':
                def get_embeddings(word_ids, we, side, word_lengths):
                    with tf.variable_scope("words"):
                        _word_embeddings = tf.Variable(we, name="_word_embeddings_" + side, dtype=tf.float32, trainable=True)
                        word_embeddings = tf.nn.embedding_lookup(_word_embeddings, word_ids, name="word_embeddings_" + side)
                    # with tf.variable_scope("chars"):
                    #     # get embeddings matrix
                    #     _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                    #                                        shape=[n_chars, char_dims])
                    #     char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids,
                    #                                              name="char_embeddings")
                    #     # put the time dimension on axis=1
                    #     s = tf.shape(char_embeddings)
                    #     char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], char_dims])
                    #     word_lengths = tf.reshape(word_lengths, shape=[-1])
                    #     # bi lstm on chars
                    #     lstm_cell = tf.contrib.rnn.LSTMCell(char_hidden_size,
                    #                                         state_is_tuple=True)
                    #     _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(lstm_cell,
                    #                                                                           lstm_cell,
                    #                                                                           char_embeddings,
                    #                                                                           sequence_length=word_lengths,
                    #                                                                           dtype=tf.float32)
                    #     output = tf.concat([output_fw, output_bw], axis=-1)
                    #     # shape = (batch size, max sentence length, char hidden size)
                    #     output = tf.reshape(output, shape=[-1, s[1], 2 * char_hidden_size])
                    #
                    #     word_embeddings = tf.concat([word_embeddings, output], axis=-1)
                    return tf.nn.dropout(word_embeddings, self.dropout_keep_prob)
                self.word_embeddings_1 = get_embeddings(self.input_x1, word_embeddings, "_X1", self.input_x1_lens)
                self.word_embeddings_2 = get_embeddings(self.input_x2, word_embeddings, "_X2", self.input_x2_lens)
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True, name="W")
                word_embeddings_1 = tf.nn.embedding_lookup(W, self.input_x1, name="word_embeddings_1")
                self.word_embeddings_1 = tf.nn.dropout(word_embeddings_1, self.dropout_keep_prob)
                word_embeddings_2 = tf.nn.embedding_lookup(W, self.input_x2, name="word_embeddings_2")
                self.word_embeddings_2 = tf.nn.dropout(word_embeddings_2, self.dropout_keep_prob)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            if use_bi_lstm:
                self.out1 = self.BiRNN(self.word_embeddings_1, self.dropout_keep_prob, "X1", self.input_x1_lens,
                                     hidden_units, number_of_layers)
                self.out2 = self.BiRNN(self.word_embeddings_2, self.dropout_keep_prob, "X2", self.input_x2_lens,
                                     hidden_units, number_of_layers)
            else:
                self.out1 = self.RNN(self.word_embeddings_1, self.dropout_keep_prob, "X1", self.input_x1_lens, hidden_units, number_of_layers)
                self.out2 = self.RNN(self.word_embeddings_2, self.dropout_keep_prob, "X2", self.input_x2_lens, hidden_units, number_of_layers)

            self.pred = self.softmax(self.out1, self.out2)
        with tf.name_scope("loss"):
            if loss_function in "mse":
                print("Using the mean squared error as loss")
                self.loss = self.mean_squared_error(self.input_y, self.pred)
            else:
                print("Using cross entropy as loss")
                self.loss = self.cross_entropy_loss(self.input_y, self.pred)
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)