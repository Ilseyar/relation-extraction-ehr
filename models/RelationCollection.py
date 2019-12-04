import json

from Entity import Entity
from RelationXu import RelationXu
from Relation import Relation


class RelationCollection:

    def __init__(self, relations):
        self.relations = relations

    @classmethod
    def from_file(cls, filepath):
        relations = []
        lines = open(filepath).readlines()
        for line in lines:
            j = json.loads(line)
            r = Relation(**j)
            r.entity1 = Entity(**r.entity1)
            r.entity2 = Entity(**r.entity2)
            relations.append(r)
        return RelationCollection(relations)

    def add_relation(self, relation):
        self.relations.append(relation)

    def add_relations(self, relations):
        self.relations.extend(relations)

    def get_relations_by_type(self, rel_type):
        result_relations = []
        for r in self.relations:
            if r.type == rel_type:
                result_relations.append(r)
        return result_relations

    def get_possible_relation_types(self):
        relation_types = set()
        for r in self.relations:
            if r.label == 1:
                relation_types.add(r.type)
        return relation_types

    def count_statistic_by_type(self):
        label_types = set()
        for r in self.relations:
            label_types.add(r.type)
        label_count = {}
        for label_type in label_types:
            label_count[label_type] = 0

        for r in self.relations:
            label_count[r.type] += 1
        # for key in label_count.keys():
        #     print(key + "\t" + "=" + "\t" + label_count[key])
        return label_count

    def count_pos_statistic_by_type(self):
        label_types = set()
        for r in self.relations:
            label_types.add(r.type)
        label_count = {}
        for label_type in label_types:
            label_count[label_type] = 0

        for r in self.relations:
            if r.label == 1:
                label_count[r.type] += 1
        return label_count

    def group_by_file(self):
        relations_by_file = {}
        for relation in self.relations:
            if relation.doc_id not in relations_by_file:
                relations_by_file[relation.doc_id] = []
            relations_by_file[relation.doc_id].append(relation)
        return relations_by_file

    def is_related(self, entity1, entity2):
        for relation in self.relations:
            if relation.entity1.id == entity1.id and relation.entity2.id == entity2.id or \
                    relation.entity1.id == entity2.id and relation.entity2.id == entity1.id:
                return True
        return False

    def print_relation_types(self):
        relation_types = set()
        for r in self.relations:
            if r.label != 0:
                relation_types.add(r.type + '\t' + r.entity1.type + "\t" + r.entity2.type)
                # relation_types.add(r.type)
                if r.type == "do" and r.entity1.type == "Dose" and r.entity2.type == "Dose":
                    print(r.toJSON())
        for relation_type in relation_types:
            print(relation_type)

    def write_to_file(self, filepath):
        f = open(filepath, 'w')
        for r in self.relations:
            f.write(r.toJSON() + "\n")
        f.close()

    def to_bioc(self):
        relations_bioc = []
        for r in self.relations:
            if r.label == 1:
                relations_bioc.append(r.to_bioc())
        return relations_bioc

    def to_ann(self):
        relation_ann = []
        for r in self.relations:
            if r.label == 1:
                relation_ann.append(r.to_ann())
        return relation_ann

    def get_labels(self):
        labels = []
        for r in self.relations:
            labels.append(r.label)
        return labels

    def set_labels(self, labels):
        assert len(labels) == len(self.relations)
        new_relations = []
        for r, l in zip(self.relations, labels):
            r.label = l
            new_relations.append(r)
        self.relations = new_relations

    def get_max_relation_id(self):
        max_id = 0
        for r in self.relations:
            if r.label == 1 and int(r.id) > max_id:
                max_id = int(r.id)
        return max_id

    def create_unique_ids(self):
        max_id = self.get_max_relation_id()
        for r in self.relations:
            if r.label == 0:
                r.id = max_id + 1
            max_id += 1

    def check_unique_id(self):
        ids = set()
        for r in self.relations:
            ids.add(r.id)
        if len(self.relations) == len(ids):
            print("All ids are unique")
        else:
            print(len(ids))
            print(len(self.relations))

    def check_empty_types(self):
        is_empty_types = False
        for r in self.relations:
            if r.type == "":
                print("type is empty")
                print(r.toJSON())
                is_empty_types = True
        if not is_empty_types:
            print("There is no empty types")

    def get_relation_types(self):
        relation_types = {}
        for r in self.relations:
            if r.label == 1:
                relation_types[r.entity1.type + "_" + r.entity2.type] = r.type
        return relation_types

    def get_positive_relations(self):
        relations = []
        for r in self.relations:
            if r.label == 1:
                relations.append(r)
        return relations

    def get_negative_relations(self):
        relations = []
        for r in self.relations:
            if r.label == 0:
                relations.append(r)
        return relations

    def get_middle_entity_types(self):
        entity_types = set()
        for r in self.relations:
            for e in r.middle_entities:
                entity_types.add(e.type)
        return list(entity_types)

    def get_relation_contexts(self):
        contexts = []
        for relation in self.relations:
            contexts.append(' '.join(relation.middle_context))
        return contexts

    @classmethod
    def from_file(cls, filepath):
        lines = open(filepath).readlines()
        relations = []
        for line in lines:
            j = json.loads(line)
            r = RelationXu.from_json(j)
            relations.append(r)
        return RelationCollection(relations)

    @classmethod
    def from_relations_doc(cls, relations, entities, text, attr_entities, attribute_candidate):
        relations_new = []
        for relation in relations:
            if relation.get_entities_distance() < RelationXu.MAX_ENTITIES_DISTANCE:
                relation_new = RelationXu.from_relation(relation, entities, text)
                if relation_new.get_entities_window_size(attr_entities, attribute_candidate) <= RelationXu.MAX_ENTITIES_WINDOW_SIZE:
                    relations_new.append(relation_new)
        return relations_new

    @classmethod
    def from_relations(cls, relation_collection, entity_collection, text_collection, attr_entities, attribute_candidate):
        relations = relation_collection.group_by_file()
        entities = entity_collection.group_by_file()
        relations_xu = []
        doc_num = len(relations.keys())
        doc_count = 1
        for doc_id in relations.keys():
            print(str(doc_count) + "/" + str(doc_num))
            text = text_collection.get_text_by_id(doc_id)
            relations_xu.extend(cls.from_relations_doc(relations[doc_id], entities[doc_id], text.text, attr_entities, attribute_candidate))
            doc_count += 1
        return RelationCollection(relations_xu)

    def toJSON(self):
        j_relations = []
        for r in self.relations:
            j_relations.append(r.toJSON())
        return j_relations
