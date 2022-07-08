from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim import utils

class MyIter(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
         path = datapath(self.path)
         with open(path, 'r', encoding='utf-8') as fin:
             for line in fin:
                 yield list(line.strip().split(" "))

path_text = "/data2/jose/projects/HTR/prepare/text_clean.txt"
epochs = 50
learning_rate = 0.025
myiter = MyIter(path_text)
res = []
for i, text in enumerate(myiter):
    res.append(text)
    # if i>10:
    #     break
# Set file names for train and test data

sizes = [300,100,50,25,10]
for size in sizes:
    fname = "fasttext/fasttext_{}epochs_{}size".format(epochs, size)
    model_gensim = FT_gensim(min_count=1, size=size, window=5, alpha=learning_rate)
    # build the vocabulary
    # print(res)
    model_gensim.build_vocab(sentences=res)
    # train the model
    model_gensim.train(
        sentences =res, epochs=epochs,
        total_examples=model_gensim.corpus_count, total_words=model_gensim.corpus_total_words
    )

    print(model_gensim)
    model_gensim.save(fname)
