import numpy as np


def read_glove_vectors(vocab, glove_filename, dim):
    """
    Saves glove vectors in numpy array
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        dim: (int) dimension of embeddings
    """
    embeddings = np.random.standard_normal((len(vocab), dim))
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = list(map(float, line[1:]))
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    return embeddings
