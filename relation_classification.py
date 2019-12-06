import json

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from features.BagOfEntitiesFeature import BagOfEntitiesFeature
from features.BagOfWordsFeature import BagOfWordsFeature
from features.BioSentenceFeature import BioSentenceFeature
from features.CharDistanceFeature import CharDistanceFeature
from features.EmbeddingFeature import EmbeddingFeature
from features.EntityTypeFeatures import EntityTypeFeature
from features.MeshSemanticTypesFeature import MeshSemanticTypesFeature
from features.PositionFeature import PositionFeature
from features.PubmedFeature import KnowledgeFeature
from features.PunctuationFeature import PunctuationFeature
from features.SentenceDistanceFeature import SentenceDistanceFeature
from features.SimilarityFeature import SimilarityFeature
from features.UMLSSemanticTypeFeature import UMLSSemanticTypeFeature
from features.WordDistanceFeature import WordDistanceFeature
from models.RelationCollection import RelationCollection
from utils import ResourceManager

ATTR_ENTITY_MADE = ""
ATTR_ENTITY_N2C2 = ""


def create_features(relations, attr_entity, attribute_candidate_types, cv, is_train, type=None, cv2=None):
    word_distance_feature = WordDistanceFeature().transform(relations)
    char_distance_feature = CharDistanceFeature().transform(relations)
    bag_of_entities_feature = BagOfEntitiesFeature(entity_types).transform(relations)
    bag_of_words_feature = BagOfWordsFeature(cv).transform(relations, cv, is_train)
    position_feature = PositionFeature(attr_entity, attribute_candidate_types).transform(relations)
    entity_type_feature = EntityTypeFeature(entity_types).transform(relations)
    sentence_distance_feature = SentenceDistanceFeature().transform(relations)
    punctuation_feature = PunctuationFeature().transform(relations)

    embedding_feature = EmbeddingFeature(w2v_model).transform(relations)
    bio_sentence_feature = BioSentenceFeature(sent2vec).transform(relations)
    similarity_feature = SimilarityFeature(w2v_model).transform(relations)
    umls_semantic_types_feature = UMLSSemanticTypeFeature(umls_semantic_types).transform(relations)
    mesh_semantic_types_feature = MeshSemanticTypesFeature(mesh_semantic_types).transform(relations)
    knowledge_feature = KnowledgeFeature(knowledge_vectors).transform(relations)

    features = numpy.concatenate((
                              word_distance_feature,
                              char_distance_feature,
                              bag_of_entities_feature,
                              bag_of_words_feature,
                              position_feature,
                              entity_type_feature,
                              sentence_distance_feature,
                              punctuation_feature,


                              embedding_feature,
                              bio_sentence_feature,
                              similarity_feature,
                              umls_semantic_types_feature,
                              mesh_semantic_types_feature,
                              knowledge_feature), axis=1)
    return features


def evaluate(gold_collection, predicted_collection):
    tp = 0
    fp = 0
    fn = 0
    for gold, pred in zip(gold_collection.relations, predicted_collection.relations):
        if gold.label == 1:
            if pred.label == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred.label == 1:
                fp += 1

    precision = float(tp)/(tp + fp)
    recall = float(tp)/(tp + fn)

    f = 2 * precision * recall / (precision + recall)

    print(str(float('{:.3f}'.format(precision))) + "\t" + str(float('{:.3f}'.format(recall))) + "\t" + str(float('{:.3f}'.format(f))))


input_train_file_name = ""
input_test_file_name = ""
w2v_model_file = ""
sent2vec_file = ""
umls_semantic_types = ""
mesh_semantic_types = ""
knowledge_vectors = ""
attr_entity_file = ""
attribute_candidate_file = ""

if __name__ == "__main__":

    relations_collection_train = RelationCollection.from_file(input_train_file_name)
    relations_collection_test = RelationCollection.from_file(input_test_file_name)

    print("Relation Collection Train Length = " + str(len(relations_collection_train.relations)))
    print("Relation Collection Test Length = " + str(len(relations_collection_test.relations)))

    w2v_model = ResourceManager.load_w2v_model(w2v_model_file)
    sent2vec = ResourceManager.load_sent2vec(sent2vec_file)
    umls_semantic_types = ResourceManager.load_semantic_types(umls_semantic_types)
    entity_types = relations_collection_train.get_middle_entity_types()
    knowledge_vectors = ResourceManager.load_knowledge_vectors()
    attr_entity = json.load(open(attr_entity_file))
    attribute_candidate_types = json.load(open(attribute_candidate_file))
    cv = CountVectorizer()
    cv2 = CountVectorizer()


    relation_types = relations_collection_train.get_possible_relation_types()
    relations_by_type_train = {}
    relations_by_type_test = {}
    for type in relation_types:
        relations_by_type_train[type] = RelationCollection(relations_collection_train.get_relations_by_type(type))
        relations_by_type_test[type] = RelationCollection(relations_collection_test.get_relations_by_type(type))

    all_predicted_labels = []
    all_gold_labels = []
    for type in relation_types:
        print("Make prediction for " + type)

        train_features = create_features(relations_by_type_train[type].relations, attr_entity,
                                         attribute_candidate_types, cv, True, type, cv2)
        print("train feature created")
        test_features = create_features(relations_by_type_test[type].relations, attr_entity, attribute_candidate_types,
                                        cv, False, type, cv2)
        print("test feature created")

        svc = RandomForestClassifier(class_weight={1: 0.7, 0: 0.3}, n_estimators=100)
        svc.fit(train_features, relations_by_type_train[type].get_labels())
        predicted_labels = svc.predict(test_features)

        all_predicted_labels.extend(predicted_labels)
        all_gold_labels.extend(relations_by_type_test[type].get_labels())

    evaluate(all_gold_labels, all_predicted_labels)

