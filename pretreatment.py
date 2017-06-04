
#coding=utf-8
import jieba
import collections as coll
import chardet
import pickle

def classifyWords(test, stopwords):

    test = test.replace("\n", "")
    test = test.replace("\r", "")
    test = test.replace(" ", "")
    seg_list = jieba.cut(test, cut_all=False)
    #stopwords = {}.fromkeys([line.rstrip() for line in open('source/stopwords')])
    final = []
    i = 0
    for seg in seg_list:
        seg = seg.encode('utf-8')
        if seg not in stopwords:
            final.append(seg)
            i += 1
    #print final
    return final


stopwords = {}.fromkeys([line.rstrip() for line in open('source/stopwords')])
output = open('source/neg.pkl', 'wb')
big_final = []
for num in range(0,1000):
    print num
    file_path = 'source/ChnSentiCorp_htl_ba_2000/neg/neg.' + bytes(num) + '.txt'
    fbos = open(file_path, 'r')
    senList = fbos.readlines()
    fbos.close()
    senListtmp = ''
    # 循环处理每一行
    for line in senList:
        fencoding = chardet.detect(line)
        if fencoding['encoding']!=None:
            try:
                senListtmp += line.decode(fencoding['encoding']).encode('utf-8')
            except Exception, e:
                continue

    final = classifyWords(senListtmp, stopwords)
    big_final.append(final)
    fbos.close()

pickle.dump(big_final, output)
output.close()