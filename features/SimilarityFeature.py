import codecs

import nltk
import numpy
import re

from sklearn.metrics.pairwise import cosine_similarity


class SimilarityFeature:

    def __init__(self, w2v_model):
        self.w2v_model = w2v_model
        pass


    def get_feature_names(self):
        return 'similarity'

    def euclid_normalized(self, v):
        return v / numpy.linalg.norm(v)

    def count_distances(self, topic_words, w2v_model, eta=0.1):
        """
        :param topic_words:
        :param w2v_model:
        :param distance: l1 and l2 proved to be the most successful in the paper
        :param eta:
        :return:
        """

        known_topic_words = [word for word in topic_words if word in w2v_model.wv]
        accum_l2 = 0
        accum_l1 = 0
        accum_cosine = 0
        accum_coord = 0

        for t0 in known_topic_words:
            for t1 in known_topic_words:
                if t0 != t1:
                    v0 = self.euclid_normalized(w2v_model.wv[t0])
                    v1 = self.euclid_normalized(w2v_model.wv[t1])

                    accum_l2 += numpy.sum((v1 - v0) ** 2)
                    accum_l1 += numpy.sum(numpy.abs(v1 - v0))
                    accum_cosine += 1 - v0.dot(v1)
                    accum_coord += numpy.sum((v0 - v1 > eta))

        # print("A total of", len(known_topic_words), "known words in topic", topic_words)

        if not len(known_topic_words):

            v0 = numpy.zeros(w2v_model.vector_size)
            v0[0] = 1
            v1 = numpy.zeros(w2v_model.vector_size)
            v1[1] = 1

            accum_l2 = numpy.sum((v1 - v0) ** 2)
            accum_l1 = numpy.sum(numpy.abs(v1 - v0))
            accum_cosine = 1 - v0.dot(v1)
            accum_coord = w2v_model.vector_size - 1

        else:
            if len(known_topic_words) > 1:
                accum_l2 /= len(known_topic_words) * (len(known_topic_words) - 1)
                accum_l1 /= len(known_topic_words) * (len(known_topic_words) - 1)
                accum_cosine /= len(known_topic_words) * (len(known_topic_words) - 1)
                accum_coord /= len(known_topic_words) * (len(known_topic_words) - 1)

        return [accum_l2, accum_l1, accum_cosine, accum_coord]


    def create_similarity_feature(self, relations):
        features = []
        for r in relations:
            words = nltk.word_tokenize(r.entity1.text)
            words.extend(nltk.word_tokenize(r.entity2.text))
            features.append(self.count_distances(words, self.w2v_model))
        return numpy.array(features)


    def fit(self):
        return self


    def transform(self, relations):
        return self.create_similarity_feature(relations)


    def fit_transform(self, relations, y=None):
        return self.create_similarity_feature(relations)