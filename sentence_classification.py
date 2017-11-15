#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import os

import gensim
import numpy
from gensim.models import word2vec
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from features.drugname_feature import load_drugname_dict, create_drugname_feature
from features.keyword_dict_feature import create_keyword_dict_feature, load_keyword_dict
from features.pmi_feature import create_pmi_feature, load_pmi_dict
from features.polarity_feature import create_polarity_feature
from features.pos_tag_feature import create_pos_tag_feature
from features.sentiment_feature import load_ru_sentilex_lexicon, create_sentiment_feature
from features.w2v_feature import create_w2v_feature


def load_data(filename):
    sentences_dict = {}
    with codecs.open(filename, "r", encoding='utf-8') as f:
        for line in f:
            line_parts = line.split("\t")
            sentence = {}
            sentence['id'] = line_parts[0]
            sentence['review_id'] = line_parts[1]
            sentence['text'] = line_parts[2]
            sentence['start'] = int(line_parts[3])
            sentence['end'] = int(line_parts[4])
            sentence['label'] = int(line_parts[5])
            sentence['review_id'] = line_parts[1]
            sentences_dict[line_parts[0]] = sentence
    with codecs.open("data/lemmatized_corpus.txt", "r", encoding='utf-8') as f:
        for line in f:
            lemmas = eval(line)
            if lemmas['sent_id'] in sentences_dict:
                sentences_dict[lemmas['sent_id']]['lemmas'] = lemmas['lemmas']
    keys = sentences_dict.keys()
    sentences = []
    for key in keys:
        sentences.append(sentences_dict[key])
    labels = []
    for s in sentences:
        #For binary classification
        # if s['label'] == 2 or s['label'] == 0 or s['label'] == 1:
        #     labels.append(1)
        # else:
        #     labels.append(0)
        #For multi-class classification
        labels.append(s['label'])
    return sentences, labels

def extract_lemmas(X):
    lemmatized_text = []
    for x in X:
        result = ""
        lemmas = x['lemmas']
        for l in lemmas:
            result += l + " "
        lemmatized_text.append(result.strip())
    return lemmatized_text

def extract_pos_lemmas(X):
    lemmas_dict = {}
    with codecs.open("data/pos_tags.txt", 'r', encoding='utf-8') as f:
        for line in f:
            line_json = eval(line)
            lemmas_dict[line_json['sent_id']] = line_json['lemmas']
    lemma_pos = []
    for x in X:
        lemmas = lemmas_dict[x['id']]
        lemmas_result = []
        for l in lemmas:
            lemmas_result.append(l["text"])
        lemma_pos.append(lemmas_result)
    return lemma_pos

def load_pos_tags(filename):
    pos_tags = {}
    with codecs.open(filename, "r", encoding='utf-8') as f:
        for line in f:
            pos_tag_json = eval(line)
            pos_tags[pos_tag_json['sent_id']] = pos_tag_json['pos']
    return pos_tags


def load_fasttext_model(filename):
    model = {}
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line_parts = line.strip().split(" ")
            vector = []
            for x in line_parts[1:]:
                vector.append(float(x))
            model[line_parts[0]] = vector
    return model

def load_polarity_dict(filename):
    polarity_dict = {}
    with codecs.open(filename, "r", encoding='utf-8') as f:
        for line in f:
            line_json = eval(line)
            if len(line_json['polarity']) != 0:
                polarity_dict[line_json['id']] = line_json['polarity'][0]['value']
            else:
                polarity_dict[line_json['id']] = "NEUTRAL"
    return polarity_dict


def create_features(vectorizer, X, is_train):
    lemmas =  extract_lemmas(X)
    if is_train:
        features = vectorizer.fit_transform(lemmas).toarray()
    else:
        features = vectorizer.transform(lemmas).toarray()
    pos_tag_feature = create_pos_tag_feature(X, pos_tags)
    lemmas_pos = extract_pos_lemmas(X)
    w2v_feature = create_w2v_feature(lemmas_pos, w2v_model)
    disease_feature = create_keyword_dict_feature(lemmas, disease_dict)
    polarity_feature = create_polarity_feature(X, polarity_dict)
    sentiment_feature = create_sentiment_feature(ru_sentilex_dict, lemmas)
    drugname_feature = create_drugname_feature(lemmas, drugname_dict)
    pmi_feature = create_pmi_feature(lemmas, pmi_dict)
    features = numpy.hstack((features, w2v_feature, disease_feature, polarity_feature, pmi_feature))
    return features


def folder_classification():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    svc = LinearSVC(class_weight='auto')
    predicted = []
    gold = []
    f_measures = []
    for i in range(1, 6):
        f_train = "data/folds/" + str(i) + "/train.txt"
        f_test = "data/folds/" + str(i) + "/test.txt"
        X_train, y_train = load_data(f_train)
        X_test, y_test = load_data(f_test)
        X_train_features = create_features(vectorizer, X_train, True)
        svc.fit(X_train_features, y_train)
        X_test_features = create_features(vectorizer, X_test, False)
        predicted_fold = svc.predict(X_test_features)
        predicted.extend(predicted_fold)
        gold.extend(y_test)
        f_measures.append(metrics.f1_score(y_test, predicted_fold, average='macro'))
    print f_measures
    print classification_report(gold, predicted, digits=3)
    print metrics.precision_score(gold, predicted, average='macro')
    print metrics.recall_score(gold, predicted, average='macro')
    print metrics.f1_score(gold, predicted, average='macro')


if __name__ == '__main__':
    pos_tags = load_pos_tags("data/resources/pos_tag_sent.txt")
    # Download from http://rusvectores.org/ru/models/
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format("data/resources/ruscorpora_1_300_10.bin", binary=True)
    disease_dict = load_keyword_dict("data/resources/disease_dict.txt")
    polarity_dict = load_polarity_dict("data/resources/polarity_sent.txt")
    # Download from http://www.labinform.ru/pub/rusentilex/
    ru_sentilex_dict = load_ru_sentilex_lexicon("data/resources/ru_sentilex.txt")

    drugname_dict = load_drugname_dict("data/resources/drugnames.txt")
    pmi_dict = load_pmi_dict("data/resources/sentiment_score_with_pmi.txt")
    folder_classification()