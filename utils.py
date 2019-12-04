import json

from gensim.models import KeyedVectors
# import sent2vec

class ResourceManager:


    @classmethod
    def load_w2v_model(cls, filepath):
        return KeyedVectors.load_word2vec_format(filepath, binary=True)

    @classmethod
    def load_sent2vec(cls, filepath):
        model = sent2vec.Sent2vecModel()
        model = model.load_model(filepath)
        print("BioSent is loaded")
        return model

    @classmethod
    def load_semantic_types(cls, filepaths):
        semantic_types = {}
        for filepath in filepaths:
            semantic_types.update(json.load(open(filepath)))
        return semantic_types

    @classmethod
    def load_mesh_semantic_types(cls, mesh_semantic_types_file):
        return json.load(open(mesh_semantic_types_file))

    @classmethod
    def load_knowledge_vectors(cls, pubmed_vectors_files1):
        data1 = json.load(open(pubmed_vectors_files1))
        return data1
