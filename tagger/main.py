import os
from tagger.data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset
from tagger.general_utils import get_logger
from tagger.model import NERModel
from tagger.config import config
from similarity import glove

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

# create dataset
train = CoNLLDataset(config.train_filename, processing_word, processing_tag, config.max_iter)
dev   = CoNLLDataset(config.dev_filename, processing_word, processing_tag, config.max_iter)
test  = CoNLLDataset(config.test_filename, processing_word, processing_tag, config.max_iter)
# get logger
logger = get_logger(config.log_path)

# build model
model = NERModel(config, embeddings, ntags=len(vocab_tags), nchars=len(vocab_chars), logger=logger)
model.build()

# train, evaluate and interact
# model.train(train, dev, vocab_tags)
# model.evaluate(test, vocab_tags)
model.interactive_shell(vocab_tags, processing_word)