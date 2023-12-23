import base64
import datetime

from django.core.handlers.wsgi import WSGIRequest
from rest_framework.decorators import action
from rest_framework_mongoengine.viewsets import ModelViewSet

from apps.drf.serializers import DataPreparationComponentSerializer, DataPreparationComponentFrameworkSerializer, \
    DatasetSerializer
from apps.model.mldr_model.data_transforming import DataPreparationComponent, DataPreparationSourceCode, DataPreparationComponentType, \
    DataPreparationComponentFramework, DataPreparationPipeline, DataPreparationComponentExecutionEnvironment, \
    DataPreparationComponentExecution, DataPreparationPipelineExecution, DatasetInstanceType, Dataset, DatasetType
from apps.crud.base_view import BaseView
from apps.model.mldr_model import OriginDataCollection
from utils.utils import get_request, json_response, get_json_r, get_user_defined_property, RequestDataType


class DataPreparationComponentFrameworkView(ModelViewSet):
    '''

    '''
    queryset = DataPreparationComponentFramework.objects.all()
    serializer_class = DataPreparationComponentFrameworkSerializer
    my_filter_fields = ('framework_name', 'framework_language', 'framework_version')

    @action(methods=['post'], detail=False)
    def add(self, request, *args, **kwargs):
        framework_name = get_json_r(request, 'framework_name', allow_none=False)
        framework_language = get_json_r(request, 'framework_language', default=None)
        framework_version = get_json_r(request, 'framework_version', default=None)

        fw = DataPreparationComponentFramework.generate_or_get(framework_name, framework_language, framework_version)
        return json_response(fw.to_dict())


'''
DataPreparationComponent
数据预处理元件
'''


class DataPreparationComponentView(ModelViewSet):
    queryset = DataPreparationComponent.objects.all()
    serializer_class = DataPreparationComponentSerializer
    my_filter_fields = ('name', 'component_type')

    def get_kwargs_for_filtering(self):
        filtering_kwargs = {}
        for field in self.my_filter_fields:
            field_value = self.request.query_params.get(field)
            if field_value:
                filtering_kwargs[field] = field_value
        return filtering_kwargs

    def get_queryset(self):
        queryset = DataPreparationComponent.objects.all()
        filtering_kwargs = self.get_kwargs_for_filtering()
        if filtering_kwargs:
            queryset = DataPreparationComponent.objects.filter(**filtering_kwargs)
        return queryset

    @action(methods=['post'], detail=False)
    def add(self, request, *args, **kwargs):
        property_list = ['name', 'component_type', 'component_parameters_list', 'framework_name', 'framework_language',
                         'framework_version', 'codes', 'file']

        name = get_request(request, 'name', allow_none=False)
        component_type = request.data.get('component_type', "")
        component_parameters_list = request.data.get('component_parameters_list', [])

        framework_name = request.data.get('framework_name', None)
        framework_language = request.data.get('framework_language', None)
        framework_version = request.data.get('framework_version', None)

        codes = request.data.get('codes', '')
        file = request.FILES.get('file', None)

        user_defined_property = get_user_defined_property(request,property_list)

        if codes == '' and file is None:
            raise ValueError('codes or file must be have one.')

        if framework_name is not None:
            fw = DataPreparationComponentFramework.generate_or_get(framework_name, framework_language,
                                                                   framework_version)
        else:
            fw = None

        if not DataPreparationComponentType.is_member(component_type):
            component_type_enum = DataPreparationComponentType.UNKNOWN
        else:
            component_type_enum = DataPreparationComponentType(component_type)

        sc = DataPreparationSourceCode.generate(f"{name}_source_code", codes=codes, file=file)

        dp = DataPreparationComponent.objects.create(name=name, component_type=component_type_enum,
                                                     component_parameters_list=component_parameters_list,
                                                     source_code=sc.id, framework=fw.id,**user_defined_property)

        return json_response(dp.to_dict())


class DataPreparationPipelineView(ModelViewSet):
    @action(methods=['post'], detail=False)
    def add(self, request, *args, **kwargs):
        name = get_json_r(request, 'name', allow_none=False)
        components = get_json_r(request, 'components', allow_none=False, allow_type=list)
        instances = []
        for c in components:
            c_instance = DataPreparationComponent.get(c)
            if c_instance is not None:
                instances.append(c_instance)

        ids = [i.id for i in instances]
        pairs = [[ids[i], ids[i + 1]] for i in range(len(ids) - 1)]

        pipline = DataPreparationPipeline.objects.create(name=name, nodes=ids, edges=pairs)

        return json_response(pipline.to_dict())


class DataPreparationComponentExecutionView(ModelViewSet):
    @staticmethod
    def handle_inputs(component_name, input_args, outputs, environment, started_time, ended_time,**kwargs):
        dp = DataPreparationComponent.get(name_origin=component_name)

        if dp is None:
            raise ValueError(f'no such pipeline:{component_name}')

        input_args_bytes = {}
        for k, v in input_args.items():
            try:
                input_args_bytes[k] = base64.b64decode(v.encode('utf-8'))
            except:
                input_args_bytes[k] = v

        outputs_bytes = None
        if outputs != '':
            try:
                outputs_bytes = base64.b64decode(outputs.encode('utf-8'))
            except:
                outputs_bytes = outputs

        env = DataPreparationComponentExecutionEnvironment.generate_from_dict(environment)

        dpe = DataPreparationComponentExecution.generate(started_time=started_time, ended_time=ended_time, dp=dp,
                                                         input_args=input_args_bytes, outputs=outputs_bytes,
                                                         env=env,**kwargs)

        return dpe

    @action(methods=['post'], detail=False)
    def add(self, request, *args, **kwargs):
        property_list = ["component_name", "input_args", "outputs", "started_time", "environment", "started_time",
                         "ended_time"]

        component_name = get_json_r(request, 'component_name', allow_none=False)
        input_args = get_json_r(request, 'input_args', default={})
        outputs = get_json_r(request, 'outputs', default='')
        environment = get_json_r(request, 'environment', default={})
        started_time = get_json_r(request, 'started_time',
                                  default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ended_time = get_json_r(request, 'ended_time', default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        start = datetime.datetime.strptime(started_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(ended_time, "%Y-%m-%d %H:%M:%S")

        user_defined_property = get_user_defined_property(request,property_list,request_type=RequestDataType.JSON)

        dpe = DataPreparationComponentExecutionView.handle_inputs(component_name, input_args, outputs, environment,
                                                                  started_time=start, ended_time=end,**user_defined_property)

        return json_response(dpe.to_dict())


class DataPreparationPipelineExecutionView(ModelViewSet):
    @action(methods=['post'], detail=False)
    def add(self, request, *args, **kwargs):
        pipeline_name = get_json_r(request, 'pipeline_name', allow_none=False)
        used_collection = get_json_r(request, 'used_collection', allow_none=False)
        executions = get_json_r(request, 'executions', allow_none=False, allow_type=list)
        started_time = get_json_r(request, 'started_time',
                                  default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ended_time = get_json_r(request, 'ended_time', default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        start = datetime.datetime.strptime(started_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(ended_time, "%Y-%m-%d %H:%M:%S")

        instances = []
        for e in executions:
            property_list = ["component_name","input_args","outputs","started_time","environment","started_time","ended_time"]
            component_name = e.get("component_name")
            input_args = e.get("input_args", {})
            output = e.get("outputs", '')
            environment = e.get("environment", {})
            started_time = e.get("started_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            ended_time = e.get("ended_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            start = datetime.datetime.strptime(started_time, "%Y-%m-%d %H:%M:%S")
            end = datetime.datetime.strptime(ended_time, "%Y-%m-%d %H:%M:%S")

            user_defined_property = {key: value for key, value in e.items() if key not in property_list}

            dpe = DataPreparationComponentExecutionView.handle_inputs(component_name, input_args, output, environment,
                                                                      started_time=start, ended_time=end,**user_defined_property)
            instances.append(dpe)

        ids = [i.id for i in instances]
        pairs = [[ids[i], ids[i + 1]] for i in range(len(ids) - 1)]

        dpp = DataPreparationPipeline.get(pipeline_name)

        oc = OriginDataCollection.get(used_collection)

        pipline = DataPreparationPipelineExecution.generate(dpp=dpp, nodes=ids, edges=pairs, started_time=start,
                                                            ended_time=end,oc=oc)

        return json_response(pipline.to_dict())


class DatasetView(BaseView):
    serializer_class = DatasetSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ('name')

    def create(self, request, *args, **kwargs):
        property_list = ["data", "file", "type", "pipeline_exe", "name","usage","features"]

        content_type = request.headers.get('Content-Type', '')

        data = None
        file = None

        if 'application/json' in content_type:
            data = get_json_r(request, 'data', allow_none=True, allow_type=dict, default={})
            file_type = get_json_r(request, 'type', allow_none=True, default=None)
            name = get_json_r(request, 'name', allow_none=True, default=None)
            pipeline_exe = get_json_r(request, 'pipeline_exe', allow_none=True, default=None)
            usage = get_json_r(request, 'usage', allow_none=True, default=DatasetType.TRAIN.value)
            features = get_json_r(request, 'features', allow_none=True, default=[])
            request_type = RequestDataType.JSON

        else:
            file = request.FILES.get('file', None)
            file_type = get_request(request, "type", default=None)
            name = get_request(request, "name", default=None)
            pipeline_exe = get_request(request, "pipeline_exe", default=None)
            usage = get_request(request, "usage", allow_none=True, default=DatasetType.TRAIN.value)
            features = get_request(request, "features", allow_none=True, default=[])
            request_type = RequestDataType.FORM_DATA

        user_defined_property = get_user_defined_property(request,property_list,request_type=request_type)

        if name is None:
            raise ValueError("name is required.")

        if pipeline_exe is None:
            raise ValueError("pipeline_exe is required.")

        if DatasetInstanceType.is_member(file_type):
            file_type = DatasetInstanceType(file_type)
        else:
            file_type = DatasetInstanceType.UNKNOWN

        if DatasetType.is_member(usage):
            usage = DatasetType(usage)
        else:
            usage = DatasetType.TRAIN

        pe = DataPreparationPipelineExecution.get(pipeline_exe)
        ds = Dataset.generate(usage=usage,name=name, file_type=file_type, pipeline_exe=pe, data=data, file=file,features=features,**user_defined_property)

        return json_response(ds.to_dict())
