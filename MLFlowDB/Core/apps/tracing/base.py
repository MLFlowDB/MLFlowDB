import os

from bson import ObjectId
from mongoengine import DoesNotExist

from apps.model.prov_dm import Artifact


class TraceGraphNode:
    def __init__(self,**kwargs):
        self.node = kwargs.get("node")
        self.type = kwargs.get("type")
        self.prov_type = kwargs.get("prov_type")
        self.object = self.get_object(self.node)
        self.uppers = []
        self.downs = []

    def get_object(self,node):
        try:
            return Artifact.objects.get(id=ObjectId(node))
        except DoesNotExist:
            raise ValueError

    def get_name(self):
        if getattr(self.object,'name',None) is not None:
            return str(getattr(self.object,'name',None))
        if getattr(self.object,'name_tmp',None) is not None:
            return str(getattr(self.object,'name_tmp',None))
        if getattr(self.object,'framework_name',None) is not None:
            return str(getattr(self.object,'framework_name',None))
        if getattr(self.object,'eval_name',None) is not None:
            return str(f"{getattr(self.object,'model_name',None)}_{getattr(self.object,'eval_name',None)}")
        if getattr(self.object,'instance_path',None) is not None:
            return os.path.basename(os.path.normpath(str(getattr(self.object,'instance_path',None))))
        return str(getattr(self.object,'create_time'))

    def _get_version(self):
        return self.object.version

    @property
    def node_with_namespace(self):
        return f"mldr::{self.get_readable_name()}"

    def get_readable_name(self):
        return f'{str(self.type).split(".")[-1]}<{str(self.get_name())}>,Version:{self._get_version()}'

    def __str__(self):
        return self.get_readable_name()

class TraceGraphEdge:
    def __init__(self,**kwargs):
        self.started = kwargs.get("started")
        self.ended = kwargs.get("ended")
        self.type = kwargs.get("type")

    def __eq__(self, other):
        if other.started == self.started and other.ended == self.ended and other.type == self.type:
            return True
        return False