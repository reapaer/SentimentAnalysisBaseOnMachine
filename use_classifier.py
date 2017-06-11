#!usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import feature
import pretreatment

def extract_features(data, best_words):
    feat = []
    for i in data:
        feat.append(feature.best_word_features(i, best_words))
    return feat


output = open('source/abc.pkl', 'wb')
stopwords = {}.fromkeys([line.rstrip() for line in open('source/stopwords')])

test2 = "非常好的酒店，四星的标准完全超值的享受，服务非常好"
big_final = []
calss = pickle.load(open('source/classifier.pkl'))
final = pretreatment.classifyWords(test2, stopwords)
big_final.append(final)

word_scores = feature.create_word_bigram_scores()
best_words = feature.find_best_words(word_scores, 1500)

pickle.dump(big_final, output)
output.close()

moto = pickle.load(open('source/abc.pkl','r')) #载入文本数据
moto_features = extract_features(moto, best_words)
pred = calss.prob_classify_many(moto_features)

for i in pred:
    print str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n'


