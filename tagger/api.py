import os
from tagger.data_utils import load_vocab, \
    get_processing_word
from tagger.general_utils import get_logger
from tagger.model import NERModel
from tagger.config import config
from similarity import glove
import time
from flask import Flask, request, jsonify

# directory for training outputs
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

# load vocabs
vocab_words = load_vocab(config.words_filename)
vocab_tags  = load_vocab(config.tags_filename)
vocab_chars = load_vocab(config.chars_filename)

# get processing functions
processing_word = get_processing_word(vocab_words, vocab_chars, lowercase=config.lowercase, chars=config.chars)
processing_tag  = get_processing_word(vocab_tags, lowercase=False)

# get pre trained embeddings
embeddings = glove.read_glove_vectors(vocab_words, config.glove_filename, 300)

# get logger
logger = get_logger(config.log_path)

app = Flask(__name__)

@app.route('/tf/api/ner/', methods=['GET'])
def ner():
    sentence = request.args.get('text')
    start_time = time.time()
    labels = model.process(vocab_tags, processing_word, sentence)
    elapsed_time = round(1000 * (time.time() - start_time))
    return jsonify(labels=labels, timeMillis=elapsed_time)


if __name__ == '__main__':
    model = NERModel(config, embeddings, ntags=len(vocab_tags), nchars=len(vocab_chars), logger=logger)
    model.build()
    model.restore()
    app.run(debug=True, use_reloader=False, port=5000)

