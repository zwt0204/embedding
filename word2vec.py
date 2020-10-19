#!/user/bin/env python
# coding=utf-8
"""
@file: word2vec.py
@author: zwt
@time: 2020/10/19 15:57
@desc: 
"""
from gensim.models import word2vec
import multiprocessing


def train_word2Vectors(sentence, embedding_size=128, window=5, min_cpunt=5):
    w2vModel = word2vec.Word2Vec(sentence, size=embedding_size, window=window, min_count=min_cpunt,
                                 workers=multiprocessing.cpu_count())
    return w2vModel


def save_model(w2vModel, word2vec_path):
    w2vModel.save(word2vec_path)


def load_model(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel


def train():
    # 如果只是一个文本文件
    sentences = word2vec.LineSentence('path')
    # 如果是多个文件
    # sentences = word2vec.PathLineSentences('path')

    word2vec_path = 'model_path'
    model = train_word2Vectors(sentences)
    save_model(model, word2vec_path)


def test():
    word2vec_path = 'model_path'
    model = load_model(word2vec_path=word2vec_path)
    res = model.wv.similarity("word1", "word2")
    print(res)