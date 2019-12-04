import json

import nltk
from bioc import BioCRelation, BioCNode

from models.Entity import Entity


class Relation:
    BOW_WINDOW_SIZE = 10

    def __init__(self, entity1, entity2, type, id, doc_id, label, context, middle_entities, left_context, right_context, middle_context):
        self.entity1 = entity1
        self.entity2 = entity2
        self.type = type
        self.id = id
        self.doc_id = doc_id
        self.label = label
        self.context = context
        self.middle_entities = middle_entities
        self.left_context = left_context
        self.right_context = right_context
        self.middle_context = middle_context

    def get_entities_distance(self):
        if self.entity1.start > self.entity2.start:
            entity1 = self.entity2
            entity2 = self.entity1
        else:
            entity1 = self.entity1
            entity2 = self.entity2
        return int(entity2.start) - int(entity1.end)

    def to_bioc(self):
        relation_bioc = BioCRelation()
        relation_bioc.id = str(self.id)
        relation_bioc.add_node(BioCNode(str(self.entity1.id), 'annotation 1'))
        relation_bioc.add_node(BioCNode(str(self.entity2.id), 'annotation 2'))
        relation_bioc.infons['type'] = self.type

        return relation_bioc

    def to_ann(self):
        if self.entity1.type == "Drug":
            id1 = self.entity2.id
            id2 = self.entity1.id
        else:
            id1 = self.entity1.id
            id2 = self.entity2.id
        return self.id + "\t" + self.type + " " + "Arg1:" + id1 + " " + "Arg2:" + id2

    MAX_ENTITIES_DISTANCE = 1000
    MAX_ENTITIES_WINDOW_SIZE = 3

    def __init__(self, relation, context, middle_entities, left_context, right_context, middle_context):
        Relation.__init__(self, relation.entity1, relation.entity2, relation.type,
                          relation.id, relation.doc_id, relation.label)
        self.context = context
        self.middle_entities = middle_entities
        self.left_context = left_context
        self.right_context = right_context
        self.middle_context = middle_context

    def get_entities_window_size(self, attr_entities, attribute_candidate):
        type = self.entity1.type + "_" + self.entity2.type
        attr_entity = ""
        if type in attr_entities:
            attr_entity = attr_entities[type]
        else:
            type = self.entity2.type + "_" + self.entity1.type
            if type in attr_entities:
                attr_entity = attr_entities[type]

        if attr_entity in attribute_candidate:
            candidate_entities = attribute_candidate[attr_entity]
        else:
            candidate_entities = []

        middle_entities = self.middle_entities
        position = 0
        for e in middle_entities:
            if e.type in candidate_entities:
                position += 1
        return position

    @classmethod
    def create_middle_entities(cls, entity1, entity2, entities):
        middle_entities = []
        start = entity1.end
        end = entity2.start
        for entity in entities:
            if entity.start > start and entity.end < end:
                middle_entities.append(entity)
        return middle_entities

    @classmethod
    def create_context(cls, text, entity1, entity2):
        if entity1.start > entity2.start:
            entity = entity1
            entity1 = entity2
            entity2 = entity
        start = int(entity1.start)
        left_context = text[max(0, start - 1000):start]
        left_words = nltk.word_tokenize(left_context)
        left_words = left_words[max(0, len(left_words) - BOW_WINDOW_SIZE):]
        left_context = ' '.join(left_words).strip()
        # print(len(left_words))

        end = int(entity2.end)
        if end != len(text) - 1:
            right_context = text[end + 1: min(end + 1000, len(text))]
            right_words = nltk.word_tokenize(right_context)
            right_words = right_words[0: min(len(right_words), 5)]
            right_context = ' '.join(right_words).strip()
        else:
            right_context = ''
            right_words = []

        middle_context = text[int(entity1.end):int(entity2.start) - 1].strip()
        middle_words = nltk.word_tokenize(middle_context)
        context = left_context + " $E1$ " + middle_context + " $E2$ " + right_context
        return context, left_words, right_words, middle_words

    @classmethod
    def from_relation(cls, relation, entities, text):
        middle_entities = Relation.create_middle_entities(relation.entity1, relation.entity2, entities)
        context, left_context, right_context, middle_context = cls.create_context(text, relation.entity1,
                                                                                  relation.entity2)
        return Relation(relation, context, middle_entities, left_context, right_context, middle_context)

    @classmethod
    def from_json(cls, relation_j):
        entity1 = Entity(**relation_j["entity1"])
        entity2 = Entity(**relation_j["entity2"])

        middle_entities = []
        for entity_j in relation_j["middle_entities"]:
            middle_entities.append(Entity(**entity_j))
        relation_xu = Relation(entity1, entity2, relation_j[type], relation_j["id"], relation_j["doc_id"], relation_j["label"],
                               relation_j["context"], middle_entities,
                               relation_j['left_context'], relation_j['right_context'], relation_j['middle_context'])
        return relation_xu

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True)



