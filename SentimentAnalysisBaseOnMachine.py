#coding=utf-8
import feature
import pickle
import nltk
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

def score(classifier, train, testSet, tag_test):
    classifier = SklearnClassifier(classifier) #在nltk 中使用scikit-learn 的接口
    classifier.train(train) #训练分类器

    pred = classifier.classify(testSet) #对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_test, pred) #对比分类预测结果和人工标注的正确结果，给出分类器准确度


dimension = ['500','1000','1500','2000','2500','3000']

file_name_pos = 'source/pos.pkl'.decode('utf-8')
file_name_neg = 'source/neg.pkl'.decode('utf-8')
pos_review = pickle.load(open(file_name_pos,'r'))
neg_review = pickle.load(open(file_name_neg,'r'))

pos = pos_review
neg = neg_review

for d in dimension:
    word_scores = feature.create_word_bigram_scores()
    best_words = feature.find_best_words(word_scores, int(d))

    posFeatures = pos_features(feature.best_word_features, pos, best_words)
    negFeatures = neg_features(feature.best_word_features, neg, best_words)

    train = posFeatures[:800]+negFeatures[:800]
    devtest = posFeatures[800:900]+negFeatures[800:900]
    test = posFeatures[900:]+negFeatures[900:]
    dev, tag_dev = zip(*devtest)

    #print 'Feature number %f' %d
    print 'BernoulliNB`s accuracy is %f' %score(BernoulliNB(), train, dev, test)
    print 'MultinomiaNB`s accuracy is %f' %score(MultinomialNB(), train, dev, test)
    print 'LogisticRegression`s accuracy is %f' %score(LogisticRegression(), train, dev, test)
    print 'SVC`s accuracy is %f' %score(SVC(), train, dev, test)
    print 'LinearSVC`s accuracy is %f' %score(LinearSVC(), train, dev, test)
    print 'NuSVC`s accuracy is %f' %score(NuSVC(), train, dev, test)
    print
