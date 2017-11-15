#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import numpy


def load_ru_sentilex_lexicon(filename):
    sentilex_dict = {}
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line_parts = line.split(",")
            sentilex_dict[line_parts[2].strip()] = line_parts[3].strip()
    return sentilex_dict

def create_sentiment_feature(sentiment_dict, lemmas):
    features = []
    punctuation = [',', '.', '!', '?', '(', ')']
    negative_words = ['не', 'нет']
    for lemma in lemmas:
        feature = [0] * 3
        score_not_zero = 0
        total_score = 0
        last_score = 0
        max_score = 0
        pos_score_in_affirmative = 0
        neg_score_in_affirmative = 0
        pos_score_in_negated = 0
        neg_score_in_negated = 0
        is_context_negated = False
        lemma = lemma.split(" ")
        for l in lemma:
            # l = unicode(l, 'utf-8')
            if l in negative_words:
                is_context_negated = True
            elif l in punctuation:
                is_context_negated = False
            if l in sentiment_dict:
                sentiment = sentiment_dict[l]
                if sentiment > max_score:
                    max_score = sentiment
                total_score += 1
                last_score = sentiment
                if sentiment != 0:
                    score_not_zero += 1
                    if is_context_negated:
                        if sentiment > 0:
                            pos_score_in_negated += 1
                        elif sentiment < 0:
                            neg_score_in_negated += 1
                    else:
                        if is_context_negated:
                            if sentiment > 0:
                                pos_score_in_affirmative += 1
                            elif sentiment < 0:
                                neg_score_in_affirmative += 1
        if score_not_zero == 0:
            score_not_zero = 1
        feature[0] = score_not_zero
        feature[1] = total_score / score_not_zero
        feature[2] = last_score
        features.append(feature)
    return features
