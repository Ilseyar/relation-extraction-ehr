import string

import nltk
import numpy
from sklearn.base import BaseEstimator


class PunctuationFeature(BaseEstimator):

    def __init__(self):
        pass

    def get_feature_names(self):
        return 'punctuation'

    def create_punctuation_feature(self, relations):
        features = []
        punctuation = string.punctuation
        for r in relations:
            middle_context = r.middle_context
            feature = 0
            for word in middle_context:
                if word in punctuation:
                    feature += 1
            features.append([feature])
        return numpy.array(features)

    def fit(self):
        return self

    def transform(self, relations):
        return self.create_punctuation_feature(relations)

    def fit_transform(self, relations, y=None):
        return self.create_punctuation_feature(relations)