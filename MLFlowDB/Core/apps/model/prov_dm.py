from datetime import datetime
from enumfields import Enum

import mongoengine as me
from bson import ObjectId
from mongoengine import Document, StringField, IntField, DoesNotExist


class Provenance:
    # Constants:
    NAMESPACE_FUNC = 'activity:'
    NAMESPACE_ENTITY = 'entity:'
    INPUT = 'input'
    OUTPUT = 'output'

    CHUNK_SIZE = 60000

class PROVObject():
    # PROV objects
    ENTITY = 'prov:entity'
    # GENERATED_ENTITY = 'prov:generatedEntity'
    # USED_ENTITY = 'prov:usedEntity'
    ACTIVITY = 'prov:activity'
    AGENT = 'prov:agent'

class PROVRelation():
    # PROV relations
    GENERATION = 'wasGeneratedBy'
    USE = 'used'
    DERIVATION = 'wasDerivedFrom'
    ATTRIBUTE = 'wasAttributeTo'
    QUOTE = 'wasQuoted'
    MEMBER = 'hadMember'
    INFORM = 'wasInformed'

class Counter(Document):
    name = StringField(required=True, unique=True)
    value = IntField(default=0)

class Artifact(me.DynamicDocument):
    meta = {'collection': 'artifact', 'allow_inheritance': True}
    artifact_type = me.StringField()
    create_time = me.DateTimeField(default=datetime.now())
    modify_time = me.DateTimeField(default=datetime.now(),null=True)
    version = IntField(default=1)

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        if self.id is None:
            self.id = ObjectId()

    def to_property_dict(self):
        res = {}
        for key,field in self._fields.items():
            data = self._data.get(key)
            res[key] = data
        return res




class Entity(Artifact):
    artifact_type = me.StringField(db_field="artifact_type",default=PROVObject.ENTITY)


class Agent(Artifact):
    artifact_type = me.StringField(db_field="artifact_type",default=PROVObject.AGENT)


class Collection(Entity):
    pass

class Activity(Artifact):
    started_at_time = me.DateTimeField(default=datetime.now(), db_field='started_at_time')
    ended_at_time = me.DateTimeField(default=datetime.now(), db_field='ended_at_time')
    artifact_type = me.StringField(db_field="artifact_type",default=PROVObject.ACTIVITY)



class Relation(me.Document):
    meta = {'collection': 'relation'}
    time = me.DateTimeField(default=datetime.now(), db_field='time')
    start_point = me.ObjectIdField(db_field='start_point')
    end_point = me.ObjectIdField(db_field='end_point')
    start_point_type = me.StringField(db_field='start_point_type')
    end_point_type = me.StringField(db_field='end_point_type')
    relation_type = me.StringField(db_field='relation_type')

    @classmethod
    def generate(cls,start_point:ObjectId, start_point_type, end_point:ObjectId,end_point_type,relation_type):
        try:
            r = Relation.objects.get(start_point=start_point, start_point_type=start_point_type, end_point=end_point,
                                end_point_type=end_point_type,
                                relation_type=relation_type)
            if r is None:
                raise DoesNotExist
            return r
        except DoesNotExist:
            return Relation.objects.create(time=datetime.now(),start_point=start_point, start_point_type=start_point_type, end_point=end_point,
                                end_point_type=end_point_type,
                                relation_type=relation_type)

    def to_dict(self):
        return {
            'time': self.time,
            'start_point': str(self.start_point),
            'end_point': str(self.end_point),
            'start_point_type': self.start_point_type,
            'end_point_type': self.end_point_type,
            'relation_type': self.relation_type
        }