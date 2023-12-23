import copy
from collections import OrderedDict
from enum import Enum, auto
from typing import List, Dict

from bson import ObjectId
from mongoengine import DoesNotExist

from apps.model.prov_dm import Relation, Artifact
from apps.tracing.base import TraceGraphNode, TraceGraphEdge
from apps.tracing.model import TracingForward


class Trace:
    def __init__(self):
        self._clear()

    def _clear(self):
        self.filter_map = {}
        self.nodes = {}
        self.edges = []
        self.visited = []
        self.except_name_list = []

    @staticmethod
    def trace_in_space(forward:TracingForward):
        pass

    def _timeline_stream(self,current_node: TraceGraphNode,forward):
        if (current_node.node, TracingForward.TIMELINE) in self.nodes:
            return
        self.visited.append((current_node.node, TracingForward.TIMELINE))
        new_current_node = copy.deepcopy(current_node)
        versioning_unique = getattr(new_current_node.object,"versioning_unique",[])
        if len(versioning_unique) <= 0:
            return
        args = {}
        for key in versioning_unique:
            args[key] = getattr(new_current_node.object,key)
        other_versions = type(new_current_node.object).objects.filter(**args)
        version_node_mapping = OrderedDict()
        for other_version_node in other_versions:
            node_id = str(other_version_node.id)

            if node_id not in self.nodes:
                graph_node = TraceGraphNode(node=str(other_version_node.id), type=other_version_node._cls,prov_type=other_version_node.artifact_type)
                self.nodes[node_id] = graph_node

            node_times = self.nodes[node_id]
            version_node_mapping[other_version_node.version] = node_times

        if len(version_node_mapping) <=1:
            return

        pairs = [(version_node_mapping[key], version_node_mapping[key + 1]) for key in sorted(version_node_mapping.keys())[:-1]]

        for node in version_node_mapping.values():
            self.visited.append((node.node, TracingForward.TIMELINE))
            if forward.is_timeline_all():
                if forward.is_upstream():
                    if (node.node, TracingForward.UP_STREAM) not in self.visited:
                        self._up_stream(node, forward)
                if forward.is_downstream():
                    if (node.node, TracingForward.DOWN_STREAM) not in self.visited:
                        self._down_stream(node, forward)

        for pair in pairs:
            old_one = pair[0]
            new_one = pair[1]

            edge = TraceGraphEdge(started=old_one.node, ended=new_one.node, type="wasPreVersionOf")

            if edge not in self.edges:
                self.edges.append(edge)


    def _up_stream(self,current_node: TraceGraphNode,forward):
        if (current_node.node,TracingForward.UP_STREAM) in self.nodes:
            return

        if current_node.type in self.except_name_list:
            return

        self.visited.append((current_node.node,TracingForward.UP_STREAM))

        new_current_node = copy.deepcopy(current_node)
        qs = Relation.objects.filter(start_point=ObjectId(current_node.node))
        qs_i = [q.to_dict() for q in qs]
        if len(qs) <= 0:
            return
        for q in qs_i:
            node_id = str(q.get("end_point"))

            if q.get("end_point_type") in self.filter_map:
                conditions = self.filter_map.get(q.get("end_point_type"))
                if node_id not in conditions:
                    continue

            if q.get("end_point_type") in self.except_name_list:
                continue

            if node_id not in self.nodes:
                try:
                    node_artifact: Artifact = Artifact.objects.get(_cls=q.get("end_point_type"), id=q.get("end_point"))
                except DoesNotExist:
                    print(q.get("end_point"))
                    continue
                node = TraceGraphNode(node=str(node_artifact.id), type=node_artifact._cls,prov_type=node_artifact.artifact_type)
                self.nodes[node_id] = node
            node_upper = self.nodes[node_id]
            node_upper.downs.append(new_current_node.node)
            new_current_node.uppers.append(node_upper.node)
            self.nodes[node_id] = node_upper
            self.nodes[current_node.node] = new_current_node

            edge = TraceGraphEdge(started=current_node.node, ended=node_upper.node,type=q.get("relation_type"))
            if edge not in self.edges:
                self.edges.append(edge)

            if forward.is_upstream():
                if (node_id, TracingForward.UP_STREAM) not in self.visited:
                    self._up_stream(node_upper, forward)
            if forward.is_downstream():
                if (node_id, TracingForward.DOWN_STREAM) not in self.visited:
                    self._down_stream(node_upper, forward)
            if forward.is_timeline():
                if (node_id, TracingForward.TIMELINE) not in self.visited:
                    self._timeline_stream(node_upper, forward)
        return

    def _down_stream(self,current_node: TraceGraphNode,forward):
        if (current_node.node,TracingForward.DOWN_STREAM) in self.nodes:
            return

        if current_node.type in self.except_name_list:
            return

        self.visited.append((current_node.node,TracingForward.DOWN_STREAM))

        new_current_node = copy.deepcopy(current_node)
        qs = Relation.objects.filter(end_point=ObjectId(current_node.node))
        qs_i = [q.to_dict() for q in qs]
        if len(qs) <= 0:
            return
        for q in qs_i:
            node_id = str(q.get("start_point"))

            if q.get("start_point_type") in self.filter_map:
                conditions = self.filter_map.get(q.get("start_point_type"))
                if node_id not in conditions:
                    continue

            if q.get("start_point_type") in self.except_name_list:
                continue

            if node_id not in self.nodes:
                try:
                    node_artifact: Artifact = Artifact.objects.get(_cls=q.get("start_point_type"), id=q.get("start_point"))
                except DoesNotExist:
                    continue
                node = TraceGraphNode(node=str(node_artifact.id), type=node_artifact._cls,prov_type=node_artifact.artifact_type)
                self.nodes[node_id] = node
            node_downer = self.nodes[node_id]
            node_downer.uppers.append(new_current_node.node)
            new_current_node.downs.append(node_downer.node)
            self.nodes[node_id] = node_downer
            self.nodes[current_node.node] = new_current_node

            edge = TraceGraphEdge(started=node_downer.node, ended=current_node.node,type=q.get("relation_type"))
            if edge not in self.edges:
                self.edges.append(edge)

            if forward.is_upstream():
                if (node_id, TracingForward.UP_STREAM) not in self.visited:
                    self._up_stream(node_downer, forward)
            if forward.is_downstream():
                if (node_id, TracingForward.DOWN_STREAM) not in self.visited:
                    self._down_stream(node_downer, forward)
            if forward.is_timeline():
                if (node_id, TracingForward.TIMELINE) not in self.visited:
                    self._timeline_stream(node_downer, forward)
        return

    def trace(self, filter_map:Dict[str,List[str]], start_point="", forward:TracingForward=TracingForward.ALL,
              except_name_list=None):
        if except_name_list is None:
            except_name_list = []
        self._clear()
        self.except_name_list = except_name_list
        self.filter_map = filter_map
        if start_point == "":
            raise NotImplementedError
        s_objs = filter_map.get(start_point)
        start_nodes = []
        for obj in s_objs:
            node_artifact: Artifact = Artifact.objects.get(id=ObjectId(obj))
            node = TraceGraphNode(node=str(obj),type=node_artifact._cls,prov_type=node_artifact.artifact_type)
            self.nodes[obj] = node
            start_nodes.append(node)
            if forward.is_upstream():
                self._up_stream(self.nodes[obj],forward)
            if forward.is_downstream():
                self._down_stream(self.nodes[obj],forward)
        return



