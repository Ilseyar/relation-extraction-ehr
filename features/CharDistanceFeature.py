import nltk
import numpy
from sklearn.base import BaseEstimator

from config import Config


class CharDistanceFeature(BaseEstimator):

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'char_distance'

    def create_char_distance_feature(self, relations):
        features = []
        for r in relations:
            features.append([len(''.join(r.middle_context))])
        # features /= numpy.max(numpy.abs(features), axis=0)
        return numpy.array(features)

    def fit(self):
        return self

    def transform(self, relations):
        return self.create_char_distance_feature(relations)

    def fit_transform(self, relations, y=None):
        return self.create_char_distance_feature(relations)