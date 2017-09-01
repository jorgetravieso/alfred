import numpy as np
import gc
from random import random
import time, os

#from preprocessing import VocabularyProcessor

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

class InputHelper(object):
    def getTsvData(self, training_paths, random_negatives=True):

        x1 = []
        x2 = []
        y = []
        chars_set = set()

        for filepath in training_paths.split(","):
            print("Loading training data from " + filepath)
            # positive samples from file
            for line in open(filepath):
                for c in line:
                    chars_set.add(c)
                # print(line)
                if len(line) == 0:
                    continue
                l = line.strip().split("\t")
                if len(l) < 2:
                    continue
                if random() > 0.5:
                    x1.append(l[0])
                    x2.append(l[1])
                else:
                    x1.append(l[1])
                    x2.append(l[0])
                y.append(1 if l[2].strip() == 'y' else 0)

        # generate random negative samples
        if random_negatives:
            combined = np.asarray(x1 + x2)
            shuffle_indices = np.random.permutation(np.arange(len(combined)))
            combined_shuff = combined[shuffle_indices]
            for i in range(len(combined)):
                x1.append(combined[i])
                x2.append(combined_shuff[i])
                y.append(0)

        return np.asarray(x1), np.asarray(x2), self.as_one_hot(y), chars_set

    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from " + filepath)
        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in open(filepath):
            if line.startswith('#'):
                continue
            l = line.strip().split("\t")
            if len(l) < 3:
                continue
            x1.append(l[1])
            x2.append(l[2])
            y.append(int(l[0]))

        return np.asarray(x1), np.asarray(x2), self.as_one_hot(y)

    def as_one_hot(self, y, n_classes=2):
        target = np.zeros(shape=(len(y), n_classes))
        for i, v in enumerate(y):
            target.itemset(i, v, 1)
        return target

    def batch_iter(self, data, batch_size, num_epochs, shuffle=False):
        """
		Generates a batch iterator for a dataset.
		"""
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                idx = batch_num * batch_size  # start_index
                result = []
                for _ in range(batch_size):
                    # print(idx)
                    result.append(data[idx])
                    idx = (idx + 1) % len(data)
                yield result

    def dumpValidation(self, x1_text, x2_text, y, shuffled_index, dev_idx, i):
        print("dumping validation " + str(i))
        x1_shuffled = x1_text[shuffled_index]
        x2_shuffled = x2_text[shuffled_index]
        y_shuffled = y[shuffled_index]
        x1_dev = x1_shuffled[dev_idx:]
        x2_dev = x2_shuffled[dev_idx:]
        y_dev = y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        with open('tmp/validation.txt' + str(i), 'w+') as f:
            for text1, text2, label in zip(x1_dev, x2_dev, y_dev):
                f.write(str(tf.argmax(label, 0)) + "\t" + text1 + "\t" + text2 + "\n")
            f.close()
        del x1_dev
        del y_dev

    # Data Preparation
    # ==================================================

    def getDataSets(self, training_paths, percent_dev, batch_size, max_length):
        x1_text, x2_text, y, char_set = self.getTsvData(training_paths, random_negatives=False)

        print("The actual max document length ", max(InputHelper.getMaxLength(x1_text), InputHelper.getMaxLength(x2_text)))
        # Build vocabulary
        print("Building vocabulary")

        vocab_processor = VocabularyProcessor(max_length)
        vocab_processor.fit(np.concatenate((x2_text, x1_text), axis=0))
        print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
        print("Length of loaded chars vocab ={}".format(len(char_set)))

        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))

        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100
        del x1
        del x2

        # Split train/test set
        self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))

        sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
        train_set = (x1_train, x2_train, y_train)
        dev_set = (x1_dev, x2_dev, y_dev)
        gc.collect()
        return train_set, dev_set, vocab_processor, sum_no_of_batches, char_set

    @staticmethod
    def getMaxLength(utterances):
        # for u in utterances:
        #	print(u, " ", len(u.split(" ")))
        def avg(elems):
            return float(sum(elems) / len(utterances))

        print("Average length", avg(len(u.split(" ")) for u in utterances))
        return max(len(u.split(" ")) for u in utterances)

    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp, x2_temp, y = self.getTsvTestData(data_path)

        # print(str(zip(x1_temp, x2_temp)) + "\n")

        # Build vocabulary
        vocab_processor = VocabularyProcessor.restore(vocab_path)
        print(len(vocab_processor.vocabulary_))

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2, y

    def restore_vocabulary_processor(self, path):
        return VocabularyProcessor.restore(path)
