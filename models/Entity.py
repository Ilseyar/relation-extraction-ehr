import json

from bioc import BioCAnnotation, BioCLocation


class Entity:
    def __init__(self, text, start, type, id, doc_id, end=None):
        self.text = text
        self.start = start
        self.type = type
        self.id = id
        self.doc_id = doc_id

        if end:
            self.end = end
        else:
            self.end = int(self.start) + len(text)

    def to_bioc(self):
        entity_bioc = BioCAnnotation()
        entity_bioc.infons['type'] = self.type
        entity_bioc.text = self.text
        entity_bioc.id = str(self.id)
        location = BioCLocation(self.start, len(self.text))
        entity_bioc.add_location(location)
        return entity_bioc

    def to_ann(self):
        return self.id + "\t" + self.type + " " + str(self.start) + \
               " " + str(self.end) + "\t" + self.text

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, skipkeys=True)

