import wget
import os.path

embeddings_file = '../embeddings/glove.6B/glove.6B.300d.txt'
if not os.path.isfile(embeddings_file):
    os.makedirs('../embeddings/glove.6B/')
    embeddings_url = 'http://learning-resources.jorgetravieso.com/embeddings/glove/glove.6B.300d.txt'
    wget.download(embeddings_url, out=embeddings_file)

from tagger.data_utils import UNK, NUM, get_glove_vocab, write_vocab

vocab_words = set()
vocab_tags = set()
vocab_chars = set()

base_path = '../private/tagger/ner/'


file = open(base_path + 'ner.combined')
for line in file:
    line = line.strip()
    if len(line) == 0:
        continue
    token, tag = line.split('_ _')
    token = token.encode('utf-8').strip()
    tag = tag.encode('utf-8').strip()
    token = token.decode("utf-8")
    tag = tag.decode("utf-8")
    print(token, tag)
    for c in token:
        vocab_chars.add(c)
    vocab_words.add(token)
    vocab_tags.add(tag)

# Build Word and Tag vocab
vocab_glove = get_glove_vocab(embeddings_file)

vocab = vocab_words & vocab_glove
vocab.add(UNK)
vocab.add(NUM)

# Save vocabs
write_vocab(vocab, base_path + "words.txt")
write_vocab(vocab_tags, base_path + "tags.txt")
write_vocab(vocab_chars, base_path + "chars.txt")