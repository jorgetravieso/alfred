

class Examples(object):
	"""
	Class that iterates over CoNLL Dataset

	__iter__ method yields a tuple (words, tags)
		words: list of raw words
		tags: list of raw tags
	If processing_word and processing_tag are not None,
	optional preprocessing is appplied

	Example:
		```python
		data = CoNLLDataset(filename)
		for sentence, tags in data:
			pass
		```
	"""

	def __init__(self, filename, processing_word=None, processing_tag=None,
				 max_iter=None):
		"""
		Args:
			filename: path to the file
			processing_words: (optional) function that takes a word as input
			processing_tags: (optional) function that takes a tag as input
			max_iter: (optional) max number of sentences to yield
		"""
		self.filename = filename
		self.processing_word = processing_word
		self.processing_tag = processing_tag
		self.max_iter = max_iter
		self.length = None

	def __iter__(self):
		niter = 0
		with open(self.filename) as f:
			words, tags = [], []
			for line in f:
				line = line.strip()
				if len(line) == 0:
					continue
				word, tag = line.split(' ')
				if self.processing_word is not None:
					word = self.processing_word(word)
				if self.processing_tag is not None:
					tag = self.processing_tag(tag)
				words += [word]
				tags += [tag]

	def __len__(self):
		"""
		Iterates once over the corpus to set and store length
		"""
		if self.length is None:
			self.length = 0
			for _ in self:
				self.length += 1

		return self.length