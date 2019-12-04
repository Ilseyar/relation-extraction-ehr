import codecs

import nltk
import numpy
import re

from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingFeature:

    def __init__(self, w2v_model):
        self.w2v_model = w2v_model
        pass


    def get_feature_names(self):
        return 'embedding'

    def get_average_vector(self, text):
        words = nltk.word_tokenize(text)
        vectors = numpy.zeros((self.w2v_model.vector_size,), dtype="float32")
        nwords = 0
        for word in words:
            if word in self.w2v_model:
                vectors = numpy.add(vectors, self.w2v_model[word])
                nwords += 1
        if nwords == 0:
            nwords = 1
        vectors = numpy.divide(vectors, nwords)
        return vectors


    def create_embedding_feature(self, relations):
        features = None
        count_non_zero = 0
        for r in relations:
            vector1 = self.get_average_vector(r.entity1.text)
            vector2 = self.get_average_vector(r.entity2.text)
            feature = numpy.concatenate((vector1, vector2), axis=0)
            if numpy.count_nonzero(vector1) != 0:
                count_non_zero += 1
            if numpy.count_nonzero(vector2) != 0:
                count_non_zero += 1
            if features is None:
                features = [feature]
            else:
                features = numpy.concatenate((features, [feature]), axis=0)
        return features


    def fit(self):
        return self


    def transform(self, relations):
        return self.create_embedding_feature(relations)


    def fit_transform(self, relations, y=None):
        return self.create_embedding_feature(relations)