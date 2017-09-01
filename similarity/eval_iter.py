#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from input_helpers import InputHelper
from time import sleep

# Parameters
# ==================================================
run_id = 'runs/1503723018'
model_file = run_id + "/checkpoints/model-72000"

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "data/test3.tsv", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", run_id + "/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", model_file, "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

if FLAGS.eval_filepath == None or FLAGS.vocab_filepath == None or FLAGS.model == None:
    print("Eval or Vocab filepaths are empty.")
    exit()

print("\nRunning iterative evaluation...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, checkpoint_file)
        inpH = InputHelper()
        vocab_processor = inpH.restore_vocabulary_processor(FLAGS.vocab_filepath)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x1_lens = graph.get_operation_by_name("input_x1_lens").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_x2_lens = graph.get_operation_by_name("input_x2_lens").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        # Tensors we want to evaluate
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        predictions = graph.get_operation_by_name("output/Softmax").outputs[0]

        def get_lengths(x_batch):
            def compute_size(x):
                last_index = None
                for i, e in enumerate(x):
                    if e is not 0:
                        last_index = i
                return len(x) if last_index is None else last_index + 1

            return [compute_size(x) for x in x_batch]


        def get_feed_dict(x1_dev_b, x2_dev_b, y_dev_b):
            return {
                input_x1: x1_dev_b,
                input_x1_lens: get_lengths(x1_dev_b),
                input_x2: x2_dev_b,
                input_x2_lens: get_lengths(x2_dev_b),
                input_y: y_dev_b,
                dropout_keep_prob: 1.0
            }

        while True:
            s1 = input("S1 >")
            s2 = input("S2 >")
            x1_test = np.asarray(list(vocab_processor.transform(np.asarray([s1]))))
            x2_test = np.asarray(list(vocab_processor.transform(np.asarray([s2]))))
            y_test = np.asarray([0.0, 1.0])

            # Fixme batch size use to be fixed to 64 in the input placeholders
            x1_test = np.tile(x1_test, (64, 1))
            x2_test = np.tile(x2_test, (64, 1))
            y_test = np.tile(y_test, (64, 1))

            batch_predictions = sess.run([predictions], get_feed_dict(x1_test, x2_test, y_test))
            print("Prediction:", batch_predictions[0][0][1])