import numpy


class UMLSSemanticTypeFeature:

    def __init__(self, semantic_types):
        self.semantic_types = semantic_types
        pass


    def get_feature_names(self):
        return 'umls_semantic_type'

    def create_umls_semantic_type_feature(self, relations):
        features = []
        uniq_semantic_types = set()
        for types in self.semantic_types.values():
            for type in types:
                uniq_semantic_types.add(type)
        uniq_semantic_types = list(uniq_semantic_types)
        size = len(uniq_semantic_types)

        count_1 = 0
        count_0 = 0
        for r in relations:
            feature = [0] * size
            # if r.type == "adverse" or r.type == 'reason':
            if r.entity1.text in self.semantic_types:
                types = self.semantic_types[r.entity1.text]
                for type in types:
                    feature[uniq_semantic_types.index(type)] += 1
            if r.entity2.type in self.semantic_types:
                types = self.semantic_types[r.entity1.text]
                for type in types:
                    feature[uniq_semantic_types.index(type)] += 1
            features.append(feature)
            if feature.count(2) > 0:
                if r.label == 1:
                    count_1 += 1
                else:
                    count_0 += 1
        print(count_0)
        print(count_1)
        return numpy.array(features)


    def fit(self):
        return self


    def transform(self, relations):
        return self.create_umls_semantic_type_feature(relations)


    def fit_transform(self, relations, y=None):
        return self.create_umls_semantic_type_feature(relations)