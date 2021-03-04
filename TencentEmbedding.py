#!/user/bin/env python
# coding=utf-8
"""
@file: TencentEmbedding.py
@author: zwt
@time: 2021/1/5 17:40
@desc: 
"""
from gensim.models import KeyedVectors
from flask import Flask, request, Response
import json
# app = Flask(__name__)
# app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
file = 'Tencent_AILab_ChineseEmbedding.txt'
wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)  # 加载时间比较长
wv_from_text.init_sims(replace=True)
#
# @app.route('/sim', methods=['POST'])
# def sim_data():
#     response = {"data": "", "returnMsg": "处理成功"}
#     data_dict = json.loads(request.get_data(as_text=True))
#     word = data_dict.get('word', '')
#     wv_from_text.init_sims(replace=True)
#     # words = ['床上四件套', 'apple', '苹果', '冬天', '手拿包', '马夹', '试用装', '半身裙', '三件套', '饺子']
#     # for word in words:
#     if word in wv_from_text.wv.vocab.keys():
#         res = []
#         vec = wv_from_text[word]
#         datas = wv_from_text.most_similar(positive=[vec], topn=10)
#         for data in datas:
#             res.append(data[0])
#     else:
#         print("没找到")
#         res = []
#     response['data'] = res
#     response = json.dumps(response, ensure_ascii=False)
#     return Response(response, mimetype='application/json')


def sim_token():
    f_w = open('title_tokens_sim.txt', 'w', encoding='utf8')
    result = []
    unknow = []
    words = []
    with open('tokens.txt', 'r', encoding='utf8') as f:
        for word in f.readlines():
            words.append(word.strip())
    for word in words:
        print('==',word)
        if word in wv_from_text.wv.vocab.keys():
            res = []
            vec = wv_from_text[word]
            datas = wv_from_text.most_similar(positive=[vec], topn=5)
            for data in datas:
                res.append(data[0])
            print(res)
            result.append(' '.join(res))
        else:
            print("没找到")
            unknow.append(word)

    for data in result:
        f_w.write(data.strip() + '\n')
    print(unknow)


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=8090, debug=True)
    sim_token()
