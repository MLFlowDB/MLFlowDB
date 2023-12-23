from rest_framework.decorators import action
from rest_framework_mongoengine.viewsets import ModelViewSet

from apps.crud.base_view import BaseView
from apps.drf.serializers import OriginDataSerializer, OriginDataCollectionSerializer
from apps.model.mldr_model.data_gathering import OriginData, OriginDataCollection
from utils.utils import get_json_r, json_response, get_user_defined_property


class OriginDataView(BaseView):
    serializer_class = OriginDataSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ()

    @action(methods=['post'], detail=False)
    def group_create(self, request, *args, **kwargs):
        ids = get_json_r(request, 'ids', allow_type=list,allow_none=False)
        res = []
        for data_id in ids:
            res.append(OriginData.get_or_generate(data_id))

        return json_response({"origin_datas":[d.to_dict() for d in res]})

class OriginDataCollectionView(BaseView):
    serializer_class = OriginDataCollectionSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ()

    def create(self, request, *args, **kwargs):
        property_list = ["name", "datas"]
        name = get_json_r(request, 'name', allow_none=False)
        datas = get_json_r(request, 'datas', default=[],allow_type=list)

        user_defined_property = get_user_defined_property(request=request, property_list=property_list)

        ds = [OriginData.get(str(data_id)) for data_id in datas]

        og = OriginDataCollection.generate(name=name, data=ds, **user_defined_property)

        return json_response(og.to_dict())