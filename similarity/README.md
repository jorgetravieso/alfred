# Deep Siamese LSTMs for Text Similarity

### Purpose
To learn a similarity measure, in the range [0, 1.0] of two arbitrary textual inputs X1 and X2

### Architecture
- Glove 300 Dim Embeddings
- Use 2 multi-layer LSTMs (1 for each of the inputs)
- The LSTMs outputs are concatenated, hence the name Siamese, into a vector `v` as follows: 
```v = tf.concat([h1, h2, tf.squared_difference(h1, h2), tf.multiply(h1, h2)], axis=-1)```
- We get the raw scores ```Wv + b``` and apply softmax<sup>[1](#note1)</sup> to it
- Softmax output (i.e. a probability) is then used a similarity measure
- Cross-entropy error is used as loss

### Example Runs
- Large Dataset<sup>[2](#note2)</sup> (400K examples, ~100K vocab)
  - 250 Hidden Size, 2 Layers, 0.001 Learning Rate
  - 84% Accuracy on test set
- Small Dataset (11K examples, 9K vocab): 
  - 35 Hidden Size 1 Layer 0.001 Learning Rate

### References
- https://web.stanford.edu/class/cs224n/reports/2748045.pdf
- http://www.aclweb.org/anthology/W16-16#page=162
- https://github.com/dhwajraj/deep-siamese-text-similarity 

#### Footnotes
<a name="note1">1.</a> Sigmoid could be used as well.
<a name="note2">2.</a> Quora Questions Pairs Dataset