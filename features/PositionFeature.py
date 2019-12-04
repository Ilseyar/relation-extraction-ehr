import numpy
from sklearn.base import BaseEstimator


class PositionFeature(BaseEstimator):

    def __init__(self, attr_entity, attribute_candidate_types):
        self.attr_entity = attr_entity
        self.attribute_candidate = attribute_candidate_types
        pass

    def get_feature_names(self):
        return 'position'

    def create_position_feature(self, relations):
        features = []
        for r in relations:
            feature = [0] * 2

            type = r.entity1.type + "_" + r.entity2.type
            attr_entity = ""
            if type in self.attr_entity:
                attr_entity = self.attr_entity[type]
            else:
                type = r.entity2.type + "_" + r.entity1.type
                if type in self.attr_entity:
                    attr_entity = self.attr_entity[type]

            if attr_entity in self.attribute_candidate:
                candidate_entities = self.attribute_candidate[attr_entity]
            else:
                candidate_entities = []

            middle_entities = r.middle_entities
            position = 0
            for e in middle_entities:
                if e.type in candidate_entities:
                    position += 1
            if r.entity1.type in self.attribute_candidate.keys():
                feature[0] = 0
                feature[1] = position
            elif r.entity2.type in self.attribute_candidate.keys():
                feature[0] = -position
                feature[1] = 0
            features.append(feature)
        return numpy.array(features)

    def fit(self):
        return self

    def transform(self, relations):
        return self.create_position_feature(relations)

    def fit_transform(self, relations, y=None):
        return self.create_position_feature(relations)