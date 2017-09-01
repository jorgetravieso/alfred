import time

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

from input_helpers import InputHelper
from util.singleton import Singleton

# Parameters
# ==================================================
run_id = 'runs/1504191876'
model_file = run_id + '/checkpoints/model-4000'

# Eval Parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
tf.flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory from training run')
tf.flags.DEFINE_string('eval_filepath', 'data/test3.tsv', 'Evaluate on this data (Default: None)')
tf.flags.DEFINE_string('vocab_filepath', run_id + '/checkpoints/vocab', 'Load training time vocabulary (Default: None)')
tf.flags.DEFINE_string('model', model_file, 'Load trained model checkpoint (Default: None)')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))

if FLAGS.eval_filepath == None or FLAGS.vocab_filepath == None or FLAGS.model == None:
    print('Eval or Vocab filepaths are empty.')
    exit()

@Singleton
class DeepTextSimilarity:

    def __init__(self):
        print('Loading the deep similarity model', FLAGS.model)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph('{}.meta'.format(FLAGS.model))
                init = tf.global_variables_initializer()
                self.sess.run(init)
                saver.restore(self.sess, FLAGS.model)
                inpH = InputHelper()
                self.vocab_processor = inpH.restore_vocabulary_processor(FLAGS.vocab_filepath)

                # Get the placeholders from the graph by name
                self.input_x1 = graph.get_operation_by_name('input_x1').outputs[0]
                self.input_x1_lens = graph.get_operation_by_name('input_x1_lens').outputs[0]
                self.input_x2 = graph.get_operation_by_name('input_x2').outputs[0]
                self.input_x2_lens = graph.get_operation_by_name('input_x2_lens').outputs[0]
                self.input_y = graph.get_operation_by_name('input_y').outputs[0]

                # Tensors we want to evaluate
                self.dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
                self.predictions = graph.get_operation_by_name('output/Softmax').outputs[0]

    def get_lengths(self, x_batch):
        def compute_size(x):
            last_index = None
            for i, e in enumerate(x):
                if e is not 0:
                    last_index = i
            return len(x) if last_index is None else last_index + 1

        return [compute_size(x) for x in x_batch]

    def get_feed_dict(self, x1_dev_b, x2_dev_b, y_dev_b):
        return {
            self.input_x1: x1_dev_b,
            self.input_x1_lens: self.get_lengths(x1_dev_b),
            self.input_x2: x2_dev_b,
            self.input_x2_lens: self.get_lengths(x2_dev_b),
            self.input_y: y_dev_b,
            self.dropout_keep_prob: 1.0
        }

    def run(self, s1, s2):
        x1_test = np.asarray(list(self.vocab_processor.transform(np.asarray([s1]))))
        x2_test = np.asarray(list(self.vocab_processor.transform(np.asarray([s2]))))
        y_test = np.asarray([0.0, 1.0])

        # Fixme batch size use to be fixed to 64 in the input placeholders
        x1_test = np.tile(x1_test, (64, 1))
        x2_test = np.tile(x2_test, (64, 1))
        y_test = np.tile(y_test, (64, 1))

        batch_predictions = self.sess.run([self.predictions], self.get_feed_dict(x1_test, x2_test, y_test))
        print('Prediction:', batch_predictions[0][0][1])
        return batch_predictions[0][0][1]


    def run_batch_one_to_many(self, text, others):
        '''
        Runs the predictions for one utterance v others
        :param text: the text
        :param others: the other utterances
        :return: 1.0 - list of distance or similarity score
        '''
        batch_size = len(others)
        x1_test = np.asarray(list(self.vocab_processor.transform(np.asarray([text]))))
        x1_test = np.tile(x1_test, (batch_size, 1))
        x2_test = np.asarray(list(self.vocab_processor.transform(others)))
        y_test = np.tile(np.asarray([0.0, 1.0]), (batch_size, 1))

        batch_predictions = self.sess.run([self.predictions], self.get_feed_dict(x1_test, x2_test, y_test))
        return [float(p[1]) for p in batch_predictions[0]]


app = Flask(__name__)

@app.route('/tf/api/similarity/', methods=['GET'])
def similarity():
    s1 = request.args.get('textA')
    s2 = request.args.get('textB')
    start_time = time.time()
    score = sim.run(s1, s2)
    elapsed_time = round(1000 * (time.time() - start_time))
    return jsonify(score=score, timeMillis=elapsed_time)


@app.route('/tf/api/similarity/batch_one_to_many', methods=['GET'])
def batch_one_to_many_similarity():
    text = request.args.get('text')
    others = request.args.getlist('others')
    start_time = time.time()
    scores = sim.run_batch_one_to_many(text, others)
    elapsed_time = round(1000 * (time.time() - start_time))
    return jsonify(scores=scores, timeMillis=elapsed_time)


if __name__ == '__main__':
    sim = DeepTextSimilarity.Instance()
    app.run(debug=True, use_reloader=False, port=5000)
