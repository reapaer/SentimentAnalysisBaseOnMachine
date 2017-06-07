#coding=utf-8
import feature
import pickle
import SentimentAnalysis
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

def pos_features(feature_extraction_method, pos, best_words):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i, best_words),'pos'] #为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method, neg, best_words):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j, best_words),'neg'] #为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures


file_name_pos = 'source/pos.pkl'.decode('utf-8')
file_name_neg = 'source/neg.pkl'.decode('utf-8')
pos_review = pickle.load(open(file_name_pos,'r'))
neg_review = pickle.load(open(file_name_neg,'r'))

pos = pos_review
neg = neg_review

word_scores = feature.create_word_bigram_scores()
best_words = feature.find_best_words(word_scores, 1500)

posFeatures = pos_features(feature.best_word_features, pos, best_words)
negFeatures = neg_features(feature.best_word_features, neg, best_words)

trainSet = posFeatures + negFeatures

MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(trainSet)
pickle.dump(MultinomialNB_classifier, open('source/classifier.pkl','w'))