
def read_stopwords(language):
    if language in "english" or language in "en":
        return read_stopwords_file("../resources/stopwords.en.list")
    raise Exception(language + " stopwords not found")


def read_stopwords_file(file):
    return set(line.strip() for line in open(file).readlines())
