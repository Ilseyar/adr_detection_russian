import codecs

import numpy

def load_drugname_dict(filename):
    drugname_dict = []
    with codecs.open(filename, 'r', encoding = 'utf-8') as f:
        for line in f:
            drugname_dict.append(line.strip())
    return drugname_dict

def create_drugname_feature(lemmas, drugname_dict):
    features = []
    for lemma in lemmas:
        count = 0
        lemma = lemma.split(" ")
        for l in lemma:
            if unicode(l, 'utf-8') in drugname_dict:
                count += 1
        feature = [count]
        features.append(feature)
    return numpy.true_divide(features, numpy.amax(features))
