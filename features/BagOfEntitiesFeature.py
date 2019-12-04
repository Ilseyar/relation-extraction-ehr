import numpy
from sklearn.preprocessing import scale


class BagOfEntitiesFeature:

    def __init__(self, entity_types):
        self.entity_types = entity_types
        pass


    def get_feature_names(self):
        return 'bag_of_entities'


    def create_bag_of_entities_feature(self, relations):
        features = []
        for r in relations:
            feature = [0] * len(self.entity_types)
            entities = r.middle_entities
            for e in entities:
                if e.type in self.entity_types:
                    feature[self.entity_types.index(e.type)] += 1
            features.append(feature)
        # features /= numpy.max(numpy.abs(features))
        return numpy.array(features)


    def fit(self):
        return self


    def transform(self, relations):
        return self.create_bag_of_entities_feature(relations)


    def fit_transform(self, relations, y=None):
        return self.create_bag_of_entities_feature(relations)