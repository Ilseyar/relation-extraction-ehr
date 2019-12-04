import numpy
from sklearn import preprocessing


class KnowledgeFeature:

    def __init__(self, pubmed_vectors):
        self.pubmed_vectors = pubmed_vectors
        pass


    def get_feature_names(self):
        return 'pubmed_vectors'

    def create_pubmed_feature(self, relations):
        for value in self.pubmed_vectors.values():
            size = len(value)
            break
        features = []
        count = 0
        max_feature = 0
        for r in relations:
            term1 = r.entity1.text + "=" + r.entity2.text
            term2 = r.entity2.text + "=" + r.entity1.text
            if term1 in self.pubmed_vectors:
                feature = self.pubmed_vectors[term1]
                count += 1
            elif term2 in self.pubmed_vectors:
                feature = self.pubmed_vectors[term2]
                count += 1
            else:
                feature = [0] * size
            if len(feature) > max_feature:
                max_feature = len(feature)
            feature = [0 if numpy.math.isnan(f) else f for f in feature]
            features.append(feature)
        new_features = features
        # for feature in features:
        #     feature = feature + [0] * (max_feature - len(feature))
        #     new_features.append(feature)
        print("Pubmed feature statistic = " + str(count))
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(numpy.array(new_features))
        features = scaler.transform(numpy.array(new_features))
        return features


    def fit(self):
        return self


    def transform(self, relations):
        return self.create_pubmed_feature(relations)


    def fit_transform(self, relations, y=None):
        return self.create_pubmed_feature(relations)