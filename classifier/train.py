import tensorflow as tf
from classifier.rnn_classifier import BiRNNClassifier
from classifier.data_reader import DataReader

# Model Hyper-Parameters
# tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
#
# # Training Parameters
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Parameters

# Network Parameters
n_classes = 2  #

dataset = DataReader('runs/1', 'data/test/sentiment.cg.train.tsv')

# Initializing the variables

model = BiRNNClassifier(sequence_length=dataset.max_document_length, num_classes=dataset.num_classes, vocab_size=dataset.vocab_size)
model.build()
model.train(dataset)