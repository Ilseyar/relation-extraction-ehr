import codecs
import json

import nltk
import numpy
import re

from sklearn.metrics.pairwise import cosine_similarity

from config import Config


class BioSentenceFeature:

    def __init__(self, sent_model=None, sent_vectors=None):
        self.sent_model = sent_model
        self.sent_vectors = sent_vectors
        pass


    def get_feature_names(self):
        return 'bio_sentence'


    def create_bio_sentence_feature(self, relations):
        features = []
        for r in relations:
            if len(r.middle_context) != 0:
                text = ' '.join(r.middle_context)
                features.append(self.sent_model.embed_sentence(text))
            else:
                features.append([0]*700)
        return numpy.array(features)

    def load_vectors_from_file(self, input_file):
        vectors = {}
        f = open(input_file)
        for line in f:
            feature = json.loads(line)
            vectors[feature['id']] = feature['sent_vec']
        return vectors

    def create_bio_sent_feature_from_file(self, relations):
        size = 700
        features = []
        count = 0
        for r in relations:
            if r.id in self.sent_vectors:
                features.append(self.sent_vectors[id])
                count += 1
            else:
                features.append([0.0] * size)
        print(count)
        print(len(relations))
        return numpy.array(features)


    def fit(self):
        return self


    def transform(self, relations):
        if self.sent_model is not None:
            return self.create_bio_sentence_feature(relations)
        elif self.sent_vectors is not None:
            return self.create_bio_sent_feature_from_file(relations)


    def fit_transform(self, relations):
        if self.sent_model is not None:
            return self.create_bio_sentence_feature(relations)
        elif self.sent_vectors is not None:
            return self.create_bio_sent_feature_from_file(relations)