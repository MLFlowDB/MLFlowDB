from typing import List

import mongoengine as me

from apps.model.prov_dm import Collection, Entity, Relation, PROVRelation
from apps.model.mldr_model import VersionedEntity
from utils.utils import get_versioning


class OriginData(VersionedEntity):
    versioning_unique = ['data_id']

    data_id = me.StringField(db_field="data_id")

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, OriginData)

    @staticmethod
    def get_or_generate(data_id):
        try:
            return OriginData.get(data_id)
        except:
            return OriginData.objects.create(data_id=data_id)

    def to_dict(self):
        return {"data_id":self.data_id}


class OriginDataCollection(VersionedEntity):
    name = me.StringField()
    origin_data_collection_ref = me.StringField(db_field="origin_data_collection_ref",null=True)
    datas = me.ListField(default=[])

    @staticmethod
    def get(name_origin, version=None):
        return get_versioning(name_origin, version, OriginDataCollection)

    @classmethod
    def generate(cls, name,data:List[OriginData], **kwargs):
        return OriginDataCollection.objects.create(name=name,datas=[d.id for d in data], **kwargs)

    def save(self, *args, **kwargs):
        used_datasets_class = [OriginData.objects.get(id=did) for did in self.datas]
        for od in used_datasets_class:
            Relation.generate(start_point=self.id, start_point_type=self._cls, end_point=od.id,
                              end_point_type=od._cls,
                              relation_type=PROVRelation.MEMBER)
        return super(OriginDataCollection, self).save(*args, **kwargs)

    def to_dict(self):
        user_dict = {}
        for k in self._dynamic_fields.to_dict().keys():
            user_dict[k] = self._data.get(k)

        defined_dict = {
            "id": str(self.id),
            "version": self.version,
            "name": self.name,
            "data_ids" : self.datas
        }
        return {**user_dict, **defined_dict}

