import numpy as np
from itertools import islice
from preprocessing.stopwords import read_stopwords


class TfIdfVectorizer:

    def __init__(self, max_document_length, tokenizer=None, stopwords=None, lowercase=False):
        self.max_document_length = max_document_length
        self.frozen = False
        self.word_doc_count = {}
        self.n_docs = 0
        self.stopwords = set()
        self.tokenizer = self.default_tokenizer
        self.lowercase = lowercase
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if stopwords is not None:
            self.stopwords = read_stopwords(stopwords)

    def freeze(self):
        self.frozen = True
        return self

    def fit(self, documents):
        if self.frozen:
            raise Exception("Cannot fit more documents as the processor is frozen")
        for d in documents:
            self.n_docs += 1
            d = list(self.apply_tokenization(d))
            for t in islice(d, self.max_document_length):
                self.word_doc_count[t] = self.word_doc_count.get(t, 0) + 1
        return self

    def transform(self, documents):
        result = np.zeros(shape=(len(documents), self.max_document_length))
        for i, d in enumerate(documents):
            d = list(self.apply_tokenization(d))
            for j, t in enumerate(d):
                if j + 1 == self.max_document_length:
                    break
                result.itemset((i, j), self.tf(t, d) * self.idf(t))
        return result

    def apply_tokenization(self, doc):
        for t in self.tokenizer(doc):
            t = t if not self.lowercase else t.lower()
            if t.lower() not in self.stopwords:
                yield t

    def default_tokenizer(self, doc):
        return [t for t in doc.split(" ")]

    def tf(self, t, d):
        return sum(1 if t is w else 0 for w in d) / len(d)

    def idf(self, t):
        return np.log(self.n_docs / (1 + self.word_doc_count.get(t, 0)))
