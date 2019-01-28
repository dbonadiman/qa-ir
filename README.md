# Question Answering and Information Retrieval

## Setup environment
You need Python 3.6. 

[Anaconda](https://www.continuum.io/downloads) is strongly recommended for the following steps.

```conda create -n qair python=3.6```

```source activate qair```

### PyTorch 1.0

```conda install pytorch torchvision -c pytorch```

If your machine support CUDA it will do the setup for you!

### Gensim 3.x

```conda install gensim```

### Spacy 2.x

```conda install -c conda-forge spacy```

```python -m spacy download en```

### Scikit-learn

```conda install -c conda-forge scikit-learn```

## Download and preprocess required data

Download the **TRECQA** and **WikiQA** dataset

Create a directory called ```original_data``` in the main directory and download the WikiQA dataset archive in it.

run the bash script:

```sh get_data.sh```

Download the embeddings of choice:

Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip

for all the models but Severyn2016 (it requires the embeddings in https://github.com/aseveryn/deep-qa)

move the embeddings into a ```embs``` directory and convert them in textual format calling them ```glove.txt``` and ```alexi.txt``` respectfully.


process the dataset the dataset

```sh process_data.sh```


## Running the model

 ```python -m qair.train configs/severyn2016.json wikiqa --name severyn_2016```

 Results are printed on screen and saved in the severyn_2016 directory

## Changelog

By now there are 3 model implemented (the original Severyn 2016 model) a CNN Baseline and a simplified version of the Relational CNN.


### More model to come ...
