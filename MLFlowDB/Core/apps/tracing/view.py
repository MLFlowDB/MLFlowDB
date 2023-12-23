import os
from enum import Enum, auto

from bson import ObjectId
from django.http import HttpResponse
from rest_framework.decorators import action
from rest_framework.viewsets import ModelViewSet

from apps.model.prov_dm import Artifact
from apps.tracing.model import ProvOnlineResult
from apps.tracing.model import TracingForward
from apps.tracing.prov_doc import ProvDoc, ProvOnline
from apps.tracing.tracing import Trace
from utils.utils import get_json_r, json_response

from apps.model.mldr_model import *

_cls_mapping = {}


class TracingResultType(Enum):
    PIC_TRACING_GRAPH = "pic"
    ONLINE_TRACING_GRAPH = "online"

    @classmethod
    def get_values(cls):
        return [member.value for member in cls]


class TracingView(ModelViewSet):

    @action(methods=['post'], detail=False)
    def trace(self, request, *args, **kwargs):
        filter_conditions = get_json_r(request, "filter_conditions", default={}, allow_none=False)
        start_point = get_json_r(request, "start_point", default="", allow_none=True)
        result_type = get_json_r(request, "result_type", allowed=TracingResultType.get_values(),
                                 default=TracingResultType.PIC_TRACING_GRAPH.value, allow_none=True)
        except_list = get_json_r(request, "except_list", default=[], allow_none=True)
        forward = get_json_r(request, "forward", allowed=TracingForward.get_values(),
                             default=TracingForward.SPACE.value, allow_none=True)

        filter_mapping = {}
        name_mapping = {}
        g = globals()
        for k, v in filter_conditions.items():
            oids = set()
            cls_name = ""

            def _handle_single_condition(cls, c):
                nonlocal cls_name
                if "_id" in c:
                    c["_id"] = ObjectId(str(c["_id"]))
                res = cls.objects.filter(**c)
                for r in res:
                    cls_name = r._cls
                    oids.add(str(r.id))

            cls = g.get(k, None)
            if cls is None:
                raise ValueError(f"No such class:{str(k)}")
            if isinstance(v, dict):
                _handle_single_condition(cls, v)
            if isinstance(v, list):
                for c in v:
                    _handle_single_condition(cls, c)
            name_mapping[k] = cls_name
            if len(oids) > 0:
                filter_mapping[cls_name] = list(oids)

        except_name_list = []
        for exp in except_list:
            exp_cls = g.get(exp, None)
            if exp_cls is None:
                raise ValueError(f"No such class:{str(exp)}")
            except_name_list.append(exp_cls()._cls)

        t = Trace()
        t.trace(filter_map=filter_mapping, start_point=name_mapping[start_point], forward=TracingForward(forward),
                except_name_list=except_name_list)

        result_type = TracingResultType(result_type)

        if result_type == TracingResultType.PIC_TRACING_GRAPH:
            image_path = ProvDoc.generate_graph(nodes=t.nodes, edges=t.edges)
            image_data = open(image_path, "rb").read()
            os.remove(image_path)
            return HttpResponse(image_data, content_type="image/png")
        elif result_type == TracingResultType.ONLINE_TRACING_GRAPH:
            res_nodes, res_edges = ProvOnline.generate_graph(nodes=t.nodes, edges=t.edges)

            r = ProvOnlineResult.objects.create(nodes=res_nodes, edges=res_edges)

            return json_response({"query_id": str(r.id)})
        return json_response({})

    @action(methods=['get'], detail=False)
    def get_value(self, request, id, *args, **kwargs):
        r = ProvOnlineResult.objects.get(id=id)
        return json_response({"nodes": r.nodes, "edges": r.edges})
