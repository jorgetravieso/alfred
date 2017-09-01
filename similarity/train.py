import tensorflow as tf
import numpy as np
import os
import time
import datetime
import wget
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from random import random
from glove import read_glove_vectors

corpus_size = 10000  # val 33281

# Parameters
# ==================================================
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "../data/similarity/train1.tsv", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 250, "Number of hidden units in softmax regression layer (default:50)")
tf.flags.DEFINE_integer("number_of_layers", 2, "Number of hidden layers for LSTM")
tf.flags.DEFINE_bool("bi_directional", False, "Whether or not to use BiDirectional LSTMS")
tf.flags.DEFINE_integer("max_document_length", 30, "The maximum size input elements (default: 30)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_float("lr_decay", 0.85, "Learning rate (default: 0.9)")
tf.flags.DEFINE_bool("use_gradient_clipping", False, "Use gradient clipping (default: False)")
tf.flags.DEFINE_integer("gradient_clipping", 5, "Gradient clipping threshold (default: 1.25)")
tf.flags.DEFINE_string("loss_function", "cross_entropy", "The loss function to use (default: cross_entropy)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("lr_decay_every", 10000, "Decay learning rate after this many steps (default: 10000)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
print("batches per epoch", int(corpus_size / FLAGS.batch_size))
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files is None:
    print("Input Files List is empty. use --training_files argument.")
    exit()

inpH = InputHelper()
train_set, dev_set, vocab_processor, sum_no_of_batches, char_set = inpH.getDataSets(FLAGS.training_files, 5, FLAGS.batch_size, max_length=FLAGS.max_document_length)

embeddings_file = '../data/glove/glove.6B/glove.6B.300d.txt'
if not os.path.isfile(embeddings_file):
    os.makedirs('../data/glove/glove.6B')
    embeddings_url = 'http://learning-resources.jorgetravieso.com/embeddings/glove/glove.6B.300d.txt'
    wget.download(embeddings_url, out=embeddings_file)

embeddings = read_glove_vectors(vocab_processor.vocabulary_._mapping, embeddings_file, 300)

# Training
# ==================================================
print("starting graph def")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    print("Started session")
    sess.run(init)
    siameseModel = SiameseLSTM(
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        hidden_units=FLAGS.hidden_units,
        number_of_layers=FLAGS.number_of_layers,
        l2_reg_lambda=FLAGS.l2_reg_lambda,
        batch_size=FLAGS.batch_size,
        word_embeddings=embeddings,
        n_chars=len(char_set),
        use_bi_lstm=FLAGS.bi_directional,
        loss_function=FLAGS.loss_function,
        max_length=FLAGS.max_document_length)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.lr_decay_every, FLAGS.lr_decay, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    print("Initialized siameseModel object")

    tr_op_set = None
    if FLAGS.use_gradient_clipping:
        gradients, variables = zip(*optimizer.compute_gradients(siameseModel.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.gradient_clipping)
        tr_op_set = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    else:
        grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
        tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("Defined training_ops")

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. TensorFlow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)

    # Print Hyper-parameters to file:
    with open(os.path.join(out_dir, "README.txt"), "w") as readme:
        for attr, value in sorted(FLAGS.__flags.items()):
            readme.write("{}={}\n".format(attr.upper(), value))
        readme.write("\n")

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Define Summary Writers
    train_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), tf.get_default_graph())
    dev_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "dev"))

    # Initialize all variables
    sess.run(tf.initialize_all_variables())

    # Merged summaries
    merged = tf.summary.merge_all()


    def train_step(x1_batch, x2_batch, y_batch):
        """
		A single training step
		"""
        feed_dict = get_fd(x1_batch, x2_batch, y_batch)
        _, step, loss, accuracy, pred, summary = sess.run(
            [tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.pred, merged],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def reverse_tokens(word_ids):
            return '\n'.join(vocab_processor.reverse(word_ids))

        for i in range(0):
            print("X1", reverse_tokens([x1_batch[i]]))
            print("X2", reverse_tokens([x2_batch[i]]))
            print(i, ", Target:", y_batch[i], ", Prediction:", '[{:g}, {:g}]'.format(pred[i][0], pred[i][1]), "\n")
        train_writer.add_summary(summary, step)
        # time.sleep()

    def get_lengths(x_batch):
        def compute_size(x):
            last_index = None
            for i, e in enumerate(x):
                if e is not 0:
                    last_index = i
            return len(x) if last_index is None else last_index + 1
        return [compute_size(x) for x in x_batch]

    def get_max_length(x_batch):
        return max(len(word_ids) for word_ids in x_batch)

    def swap(a, b):
        return b, a

    def get_fd(x1_batch, x2_batch, y_batch, dropout=FLAGS.dropout_keep_prob):
        if random() < 0.5:
            x1_batch, x2_batch = swap(x1_batch, x2_batch)
        return {
            siameseModel.input_x1: x1_batch,
            siameseModel.input_x1_lens: get_lengths(x1_batch),
            siameseModel.input_x2: x2_batch,
            siameseModel.input_x2_lens: get_lengths(x2_batch),
            siameseModel.input_y: y_batch,
            siameseModel.dropout_keep_prob: dropout,
        }

    def dev_step(x1_batch, x2_batch, y_batch):
        """
			A single training step
		"""
        step, loss, accuracy, pred, summary = sess.run(
            [global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.pred, merged],
            get_fd(x1_batch, x2_batch, y_batch, dropout=1.0))
        time_str = datetime.datetime.now().isoformat()
        print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        dev_writer.add_summary(summary, step)
        return accuracy


    # Generate batches
    batches = inpH.batch_iter(
        list(zip(train_set[0], train_set[1], train_set[2])), FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)

    ptr = 0
    max_validation_acc = 0.0
    for nn in range(sum_no_of_batches * FLAGS.num_epochs):
        batch = next(batches)
        if len(batch) < 1:
            continue
        x1_batch, x2_batch, y_batch = zip(*batch)
        if len(y_batch) < 1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        counter = 0
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db) < 1:
                    continue
                x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
                if len(y_dev_b) < 1:
                    continue
                counter += 1
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
            print("Averaged batch accuracy was: ", sum_acc / counter)
        time.sleep(2)
        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(nn) + ".pb",
                                     as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc,
                                                                                      checkpoint_prefix))
            else:
                print("Ignoring the current model as the accuracy didn't improve")
