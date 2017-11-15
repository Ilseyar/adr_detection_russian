import codecs
import re

import numpy as np

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    # index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        word = unicode(word, 'utf-8')
        if word in model:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    # print featureVec,nwords,words
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    wout = codecs.open("PubMed-and-PMC-w2v_mesh_words.txt", "w", encoding="utf-8")

    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews

    # clean_train_reviews = []
    # for w in words:
    #     w = re.sub("\+", " ", w)
    #     clean_train_reviews.append(w.split())

    for review in reviews:
        clean_train_reviews = []
        w = re.sub("\+", " ", review)
        clean_train_reviews = [w.split()]

        for word in clean_train_reviews:
            #
            # Print a status message every 1000th review
            if counter % 1000. == 0.:
                print "Review %d of %d" % (counter, len(reviews))
            #
            # Call the function (defined above) that makes average feature vectors
            vec1 = makeFeatureVec(word, model,
                                       num_features)
            reviewFeatureVecs[counter] = vec1

        vec1 = [str(l) for l in vec1.tolist()]
        print vec1

        wout.write(review + "\t" + "\t".join(vec1) + "\n")

        #
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def create_w2v_feature(lemmas, model):
    features = []
    for l in lemmas:
        feature = makeFeatureVec(l, model, model.vector_size)
        feature_clear = []
        for x in feature:
            if np.math.isnan(x):
                feature_clear.append(0.0)
            else:
                feature_clear.append(x)
        features.append(feature_clear)
    return features