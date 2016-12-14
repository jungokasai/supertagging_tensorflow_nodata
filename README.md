# A TensorFlow implementation of TAG Supertagging 

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 0.10 and 0.11 are supported.
If you want to use [Word2Vec](https://code.google.com/archive/p/word2vec/) embedding vectors, [gensim](https://radimrehurek.com/gensim/) also needs to be installed. 

## Downloading Embedding Vectors

Our architecture utilizes pre-trained word embedding vectors, [GloveVectors](http://nlp.stanford.edu/projects/glove/) and [Word2Vec](https://code.google.com/archive/p/word2vec/). For [GloveVectors](http://nlp.stanford.edu/projects/glove/), run
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip 
```
and save it to a sub-directory glovevector/. 

For [Word2Vec](https://code.google.com/archive/p/word2vec/), download GoogleNews-vectors-negative300.bin.gz and save it to a sub-directory word2vec/.  


```bash
python tf_pos_lstm_main.py train -T Super_models 
```

## Data Split

- seciton wise split
- sentence wise random split 



## Training the Vanilla Network

In order to train the network, execute
```bash
python tf_pos_lstm_main.py train 
```

You can see documentation on each of the training configurations by running
```bash
python tf_pos_lstm_main.py --help
```
## Jackknife POS tagging for Supertagging Training

We also implemented the jackknife training for Supertagging. 
For the details, see Paper.
For the jackknife training, you first need to train a POS tagging network ont the entire training set. Run, for instance, 
```bash
python tf_pos_lstm_main.py train -w 1 -u 128 
```
Then, to get an array of predicted tags on the test set and store it into ../k_fold_sec/, run the trained network
```bash
python tf_pos_lstm_main.py test -d ../POS_models/POS_tagging_cap1_num1_bi1_numlayers2_embeddim100_embedtypeglovevector_seqlength-1_seed0_longskip1_units128_lrate0.01_normalize0_dropout1.0_inputdp1.0_embeddingtrain1_suffix1_windowsize1 -m epoch6_accuracy0.96974_unknown0.91062.weights 
``` 
Then, run the k_fold jackknife. Execute, for example,
```bash
python tf_pos_lstm_main.py train -T Super_models -u 256 -w 3 -p 0.5 -i 0.8
```

## Residual Networks 


## Training the Attention model


## Test (Additinal Training) Option

You can also test an existing model. 

## Notes on the Implementation

TensorFlow has built-in functions for LSTMs. However, I have decided to explicitly code up the parameters and LSTM functions so it is clear where we apply dropout and easy to extract weights if necessary.   



