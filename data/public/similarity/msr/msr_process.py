"""

Script to read the microsoft research paraphrase corpus.
  - https://github.com/hohoCode/textSimilarityConvNet/tree/master/data/msrvid
"""


def read_file(path):
    """

    :param path: the path to the file
    :return: a list of tuples (sentence A, sentence B, label)
    """
    lines = []
    for line in open(path):
        line = line.strip()
        if line.startswith('#'):  # skip comments
            continue
        split = line.split('\t')
        label = 'y' if float(split[3]) > 2.50 else 'n'
        lines.append((split[1], split[2], label))
    return lines


train_file = 'original/msr.train.txt'
test_file = 'original/msr.test.txt'

out = open('msr.train', 'wb')
for line in read_file(train_file):
    print(line)
    out.write(bytes(line[0] + '\t' + line[1] + '\t' + line[2] + '\n', 'UTF-8'))

out.flush()
out.close()
