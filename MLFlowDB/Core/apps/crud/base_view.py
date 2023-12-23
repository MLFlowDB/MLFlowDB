from rest_framework_mongoengine.viewsets import ModelViewSet
import mongoengine as me

from utils.utils import json_response


class EmptyModel(me.Document):
    pass

class BaseView(ModelViewSet):
    cls = EmptyModel
    queryset = cls.objects.all()
    serializer_class = None
    my_filter_fields = []

    def get_kwargs_for_filtering(self):
        filtering_kwargs = {}
        for field in self.my_filter_fields:
            field_value = self.request.query_params.get(field)
            if field_value:
                filtering_kwargs[field] = field_value
        return filtering_kwargs

    def get_queryset(self):
        filtering_kwargs = self.get_kwargs_for_filtering()
        if filtering_kwargs:
            queryset = self.cls.objects.filter(**filtering_kwargs)
        else:
            queryset = self.cls.objects.all()
        return queryset

    def list(self, request, *args, **kwargs):
        return json_response([q.to_dict() for q in self.get_queryset()])