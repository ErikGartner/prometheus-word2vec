# Prometheus word2vec
*This repository contains the code to generate the word2vec model for Prometheus.*

## Setup
Checkout the word2vec submodule (the [reference implementation](https://github.com/tmikolov/word2vec)) and then build it.
```bash
git submodule init
cd word2vec
make
```

## Training
We use the following plaintext corpora:

- [sv](https://www.dropbox.com/sh/r2hu8qo281u25n7/AABv2dpqtpnNIr7BAgPoloYja?dl=0) - Vocab size: 1163288 Words in train file: 284410463
- [en](https://www.dropbox.com/sh/lgzu8a90a0fvkl8/AAC_JbCXuOvuJMu7FTD5nnw9a?dl=0) - Vocab size: 4891175, Words in train file: 2989787812

To allow for an unknown vector first create the vocabulary, manually append it a
unknown word and then train the model.

```bash
./word2vec -train <input.txt> -save-vocab vocab.txt
echo "__UNKNOWN__ 0" >> vocab.txt
```

```bash
./word2vec -train <input.txt> -binary 1 -output <model.bin> -size 300 -window 5 -sample 1e-4 -negative 5 -hs 0 -cbow 1 -iter 3 -read-vocab vocab.txt -threads 4
```
