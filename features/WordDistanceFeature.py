import nltk
import numpy
from sklearn.base import BaseEstimator

from config import Config


class WordDistanceFeature(BaseEstimator):

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'word_distance'

    def create_word_distance_feature(self, relations):
        features = []
        max = 1
        for r in relations:
            features.append([len(r.middle_context)])
            if len(r.middle_context) > max:
                max = len(r.middle_context)
        # features /= numpy.max(numpy.abs(features), axis=0)
        return numpy.array(features)

    def fit(self):
        print("Fit called")
        return self

    def transform(self, x):
        print("Transform called")
        return self.create_word_distance_feature(x)

    def fit_transform(self, x):
        print("Fit transform called")
        return self.create_word_distance_feature(x)