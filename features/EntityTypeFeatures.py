import numpy


class EntityTypeFeature:

    def __init__(self, entity_types):
        self.entity_types = entity_types
        pass


    def get_feature_names(self):
        return 'entity_type'


    def create_entity_type_feature(self, relations):
        features = []
        for r in relations:
            feature = [0] * len(self.entity_types)
            if r.entity1.type in self.entity_types:
                feature[self.entity_types.index(r.entity1.type)] += 1
            if r.entity2.type in self.entity_types:
                feature[self.entity_types.index(r.entity2.type)] += 1
            features.append(feature)
        return numpy.array(features)


    def fit(self):
        return self


    def transform(self, relations):
        return self.create_entity_type_feature(relations)


    def fit_transform(self, relations, y=None):
        return self.create_entity_type_feature(relations)