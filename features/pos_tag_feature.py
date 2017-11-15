import numpy


def create_pos_tag_feature(sentences, pos_tags):
    features = []
    for sentence in sentences:
        adverb = 0
        subj = 0
        verb = 0
        adj = 0
        sent_id = sentence['id']
        poses = pos_tags[sent_id]
        for p in poses:
            if p == "A":
                adverb += 1
            if p == "S":
                subj += 1
            if p == "V":
                verb += 1
            if p == "ADJ":
                adj += 1
        feature = [adverb, subj, verb, adj]
        features.append(feature)
    return numpy.true_divide(features, numpy.amax(features))
    # return features/numpy.amax(features)

