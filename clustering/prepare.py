import nltk

def tokenized(str):
    return ' '.join(nltk.sent_tokenize(str))

with open("utterances2.txt") as in_file:
    with open("train2.txt", "w+") as out_file:
        for line in in_file.readlines():
            out_file.write(tokenized(line.strip()))

in_file.close()
out_file.flush()
out_file.close()
