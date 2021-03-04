#!/user/bin/env python
# coding=utf-8
"""
@file: doc2vec.py
@author: zwt
@time: 2020/10/19 16:08
@desc: 
"""
import gensim.models as g
from gensim.models.doc2vec import LabeledSentence


# doc2vec parameters
vector_size = 100
window_size = 1
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0  # 0 = dbow; 1 = dmpv 类似与sk
worker_count = 1  # number of parallel processes
hs = 0  # 0 负采样，1 层次softmax


def train():
    train_corpus = 'data/query_data.txt'
    save_path = 'model/doc2vec.bin'
    docs = g.doc2vec.TaggedLineDocument(train_corpus)
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
                      workers=worker_count, hs=hs, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)
    model.save(save_path)


def demo():
    m = g.Doc2Vec.load("model/doc2vec.bin")
    data = ['七彩面膜']

    res = m.most_similar(data)
    print(m['七彩面膜'])
    # for line in data:
    #     res = m.most_similar(line.strip().split())
    #     print(res)


def labelize_reviews(reviews, label_type='SENTENCE'):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


def train_list(data_list):
    train_data = labelize_reviews(data_list)
    model = g.Doc2Vec(min_count=1, window=5, vector_size=30, sample=1e-3, negative=5, workers=3)
    model.build_vocab(train_data)
    model.train(train_data, epochs=10, total_examples=model.corpus_count)
    model.save(fname_or_handle="doc_desc")


if __name__ == '__main__':
    # train()
    demo()