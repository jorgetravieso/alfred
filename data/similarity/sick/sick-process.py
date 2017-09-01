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
        print(split)
        label = 'y' if float(split[3]) > 2.50 else 'n'
        lines.append((split[1], split[2], label, split[3]))
    return lines


file = 'original/SICK_train.txt'

out = open('sick.train', 'wb')
for line in read_file(file):
    print(line)
    out.write(bytes(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\n', 'UTF-8'))

out.flush()
out.close()
