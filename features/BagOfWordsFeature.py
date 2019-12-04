import numpy


class BagOfWordsFeature:

    def __init__(self, count_vectorizer):
        self.cv = count_vectorizer
        pass


    def get_feature_names(self):
        return 'bag_of_words'


    def create_bag_of_words_feature(self, relations, cv, is_train):
        texts = []
        for r in relations:
            text = ""
            text += ' '.join(r.left_context)
            text += r.entity1.text
            text += ' '.join(r.middle_context[0:min(5, len(r.middle_context))])
            text += ' '.join(r.middle_context[max(0, len(r.middle_context) - 5):])
            text += r.entity2.text
            text += ' '.join(r.right_context)
            texts.append(text)
        if is_train:
            X = cv.fit_transform(texts)
        else:
            X = cv.transform(texts)
        X = X.toarray()
        # X = list(X)
        # X /= numpy.max(numpy.abs(X), axis=0)
        return X


    def fit(self):
        return self


    def transform(self, relations, cv, is_train):
        return self.create_bag_of_words_feature(relations, cv, is_train)


    def fit_transform(self, relations, cv, is_train):
        return self.create_bag_of_words_feature(relations, cv, is_train)