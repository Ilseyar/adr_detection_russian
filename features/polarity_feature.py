def create_polarity_feature(sentences, polarity):
    polarity_dict = {
        'POSITIVE': 1,
        'NEGATIVE': -1,
        'NEUTRAL': 0
    }
    features = []
    for sentence in sentences:
        id = sentence['id']
        feature = [polarity_dict[polarity[id]]]
        features.append(feature)
    return features