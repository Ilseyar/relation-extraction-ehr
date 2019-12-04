import nltk
import numpy
from sklearn.base import BaseEstimator


class SentenceDistanceFeature(BaseEstimator):

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'sentence_distance'

    def create_sentence_distance_feature(self, relations):
        features = []
        for r in relations:
            features.append([len(nltk.sent_tokenize(' '.join(r.middle_context)))])
        # features /= numpy.max(numpy.abs(features), axis=0)
        return numpy.array(features)

    def fit(self):
        return self

    def transform(self, relations):
        return self.create_sentence_distance_feature(relations)

    def fit_transform(self, relations, y=None):
        return self.create_sentence_distance_feature(relations)