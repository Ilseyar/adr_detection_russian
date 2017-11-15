import codecs

import numpy

def load_keyword_dict(filename):
    keyword_dict = []
    with codecs.open(filename, "r", encoding='utf-8') as f:
        for line in f:
            keyword_dict.append(line.strip())
    return keyword_dict

def create_keyword_dict_feature(lemmas, keyword_dict):
    count = 0
    features = []
    for lemma in lemmas:
        count = 0
        lemma = lemma.split(" ")
        for l in lemma:
            if unicode(l, 'utf-8') in keyword_dict:
                count += 1
        feature = [count]
        features.append(feature)
    return numpy.true_divide(features, numpy.amax(features))