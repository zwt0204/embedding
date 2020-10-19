#!/user/bin/env python
# coding=utf-8
"""
@file: doc2vec.py
@author: zwt
@time: 2020/10/19 16:08
@desc: 
"""
import gensim.models as g

# doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0  # 0 = dbow; 1 = dmpv 类似与sk
worker_count = 1  # number of parallel processes
hs = 0  # 0 负采样，1 层次softmax


def train():
    train_corpus = 'path'
    save_path = 'model_path'
    docs = g.doc2vec.TaggedLineDocument(train_corpus)
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
                      workers=worker_count, hs=hs, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)
    model.save(save_path)


def test():
    m = g.Doc2Vec.load("model_path")
    data = ['分 词 后 的 句 子 列 表']
    for line in data:
        res = m.infer_vector(line.strip().split())
        print(res)
