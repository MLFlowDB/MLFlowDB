import datetime

from rest_framework.decorators import action
from rest_framework_mongoengine.viewsets import ModelViewSet

from apps.crud.base_view import BaseView
from apps.drf.base_serializer import UserDefinedMeasureSerializer, UserDefinedAlgorithmClassSerializer, \
    UserDefinedToolSerializer
from apps.drf.serializers import MachineLearningImplementationSerializer, MachineLearningTaskSerializer, \
    MachineLearningTrainingSerializer, FeatureSelectionSerializer, MachineLearningModelSerializer, \
    MachineLearningModelInstanceSerializer, MachineLearningEvaluationSerializer, \
    MachineLearningEvaluationExecutionSerializer
from apps.model.mex import MachineLearningProblemType, MachineLearningMethodType, MachineLearningAlgorithmClassType, \
    MachineLearningAlgorithmClass, UserDefinedMeasure, UserDefinedAlgorithmClass, UserDefinedTool
from apps.model.mldr_model.data_transforming import Dataset
from apps.model.mldr_model.ml_activity import MachineLearningFramework, MachineLearningSourceCode, \
    MachineLearningImplementation, MachineLearningTask, MachineLearningTraining, MachineLearningExecutionEnvironment, \
    FeatureSelection, MachineLearningModel, MachineLearningModelInstance, MachineLearningEvaluation, \
    MachineLearningEvaluationExecution, MachineLearningEvaluationResult
from utils.utils import get_json_r, json_response, get_request, get_user_defined_property, RequestDataType


class UserDefinedMeasureView(BaseView):
    cls = UserDefinedMeasure
    queryset = cls.objects.all()
    serializer_class = UserDefinedMeasureSerializer
    my_filter_fields = ('name', 'formula', 'measuring_class')

    def get_kwargs_for_filtering(self):
        return super(UserDefinedMeasureView, self).get_kwargs_for_filtering()

    def get_queryset(self):
        return super(UserDefinedMeasureView, self).get_queryset()


class UserDefinedAlgorithmClassView(BaseView):
    cls = UserDefinedAlgorithmClass
    queryset = cls.objects.all()
    serializer_class = UserDefinedAlgorithmClassSerializer
    my_filter_fields = ('name')

    def get_kwargs_for_filtering(self):
        return super(UserDefinedAlgorithmClassView, self).get_kwargs_for_filtering()

    def get_queryset(self):
        return super(UserDefinedAlgorithmClassView, self).get_queryset()


class UserDefinedToolView(BaseView):
    cls = UserDefinedTool
    queryset = cls.objects.all()
    serializer_class = UserDefinedToolSerializer
    my_filter_fields = ('name')

    def get_kwargs_for_filtering(self):
        return super(UserDefinedToolView, self).get_kwargs_for_filtering()

    def get_queryset(self):
        return super(UserDefinedToolView, self).get_queryset()


class MachineLearningFrameworkView(ModelViewSet):
    @action(methods=['post'], detail=False)
    def add(self, request, *args, **kwargs):
        framework_name = get_json_r(request, 'framework_name', allow_none=False)
        framework_language = get_json_r(request, 'framework_language', default=None)
        framework_version = get_json_r(request, 'framework_version', default=None)

        fw = MachineLearningFramework.generate_or_get(framework_name, framework_language, framework_version)
        return json_response(fw.to_dict())


class MachineLearningImplementationSourceCodeView(ModelViewSet):
    @action(methods=['post'], detail=False)
    def add(self, request, *args, **kwargs):
        name = get_request(request, 'name', allow_none=False)
        file = request.FILES.get('file', None)

        mi: MachineLearningImplementation = MachineLearningImplementation.get(name)

        sc_name = f"{name}_source_code"
        sc = MachineLearningSourceCode.generate(name=sc_name, file=file)

        mi.machine_learning_algorithm_source = sc.id
        mi.save()

        return json_response(mi.to_dict())


class MachineLearningImplementationView(BaseView):
    cls = MachineLearningImplementation
    queryset = cls.objects.all()
    serializer_class = MachineLearningImplementationSerializer
    my_filter_fields = ('name')

    def create(self, request, *args, **kwargs):
        property_list = ["name", "learning_method", "learning_problem", "algorithm_class", "framework_name",
                         "framework_language", "framework_version"]

        name = get_json_r(request, 'name', allow_none=False)

        learning_method = get_json_r(request, 'learning_method', allow_none=True,
                                     allowed=MachineLearningMethodType.get_values(),
                                     default=MachineLearningMethodType.UNKNOWN.value)
        learning_problem = get_json_r(request, 'learning_problem', allow_none=True,
                                      allowed=MachineLearningProblemType.get_values(),
                                      default=MachineLearningProblemType.UNKNOWN.value)
        algorithm_class = get_json_r(request, 'algorithm_class', allow_none=True,
                                     default=[])

        ac_list = []

        for ac in algorithm_class:
            if not MachineLearningAlgorithmClassType.is_member(ac):
                raise ValueError(f"wrong algorithm class:{ac}")
            ac_list.append(MachineLearningAlgorithmClassType(ac))

        status, res = MachineLearningAlgorithmClass.check_disjoint(ac_list)
        if not status:
            raise ValueError(res)

        framework_name = get_json_r(request, 'framework_name', default=None)
        framework_language = get_json_r(request, 'framework_language', default=None)
        framework_version = get_json_r(request, 'framework_version', default=None)

        user_defined_property = get_user_defined_property(request=request, property_list=property_list)

        fw = MachineLearningFramework.generate_or_get(framework_name, framework_language, framework_version)

        mi = MachineLearningImplementation.generate(name=name, learning_problem=learning_problem,
                                                    learning_method=learning_method, algorithm_class=algorithm_class,
                                                    framework=fw, **user_defined_property)

        return json_response(mi.to_dict())


class MachineLearningTrainingView(BaseView):
    cls = MachineLearningTraining
    queryset = cls.objects.all()
    serializer_class = MachineLearningTrainingSerializer
    my_filter_fields = ('task')

    def create(self, request, *args, **kwargs):
        property_list = ["task_name", "environment", "hyper_parameter", "hyper_parameter", "used_datasets", "imp_name",
                         "started_time", "ended_time"]
        task_name = get_json_r(request, 'task_name', allow_none=False)
        imp_name = get_json_r(request, 'imp_name', allow_none=False)
        environment = get_json_r(request, 'environment', allow_none=True, default={})
        hyper_parameter = get_json_r(request, 'hyper_parameter', allow_none=False, default={})
        used_datasets = get_json_r(request, 'used_datasets', allow_none=True, default=[])

        started_time = get_json_r(request, 'started_time',
                                  default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ended_time = get_json_r(request, 'ended_time', default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        start = datetime.datetime.strptime(started_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(ended_time, "%Y-%m-%d %H:%M:%S")

        env = MachineLearningExecutionEnvironment.generate_from_dict(environment)

        used_datasets_class = [Dataset.get(d_name) for d_name in used_datasets]
        user_defined_property = get_user_defined_property(request=request, property_list=property_list)

        return json_response(
            MachineLearningTraining.generate(hyper_parameter, task_name=task_name, env=env, imp_name=imp_name,
                                             used_datasets=used_datasets_class,
                                             started_time=start, ended_time=end, **user_defined_property).to_dict())


class MachineLearningTaskView(BaseView):
    cls = MachineLearningTask
    queryset = cls.objects.all()
    serializer_class = MachineLearningTaskSerializer
    my_filter_fields = ('name')


class FeatureSelectionView(BaseView):
    serializer_class = FeatureSelectionSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ('name')

    def create(self, request, *args, **kwargs):
        name = get_json_r(request, 'name', allow_none=False)
        input_datasets = get_json_r(request, 'input_datasets', allow_none=False)
        output_datasets = get_json_r(request, 'output_datasets', allow_none=False)
        selected_features = get_json_r(request, 'selected_features', allow_type=dict, allow_none=True, default={})
        started_time = get_json_r(request, 'started_time',
                                  default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ended_time = get_json_r(request, 'ended_time', default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        start = datetime.datetime.strptime(started_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(ended_time, "%Y-%m-%d %H:%M:%S")

        input_datasets_class = [Dataset.get(d_name) for d_name in input_datasets]
        output_datasets_class = [Dataset.get(d_name) for d_name in output_datasets]

        return json_response(
            FeatureSelection.generate(name, input_datasets_class, output_datasets_class,
                                      selected_features=selected_features, started_time=start,
                                      ended_time=end).to_dict())


class MachineLearningModelInstanceView(BaseView):
    serializer_class = MachineLearningModelInstanceSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ()

    def create(self, request, *args, **kwargs):
        name = get_request(request, 'name', allow_none=False)
        file = request.FILES.get('file', None)

        model: MachineLearningModel = MachineLearningModel.get(name)

        mi = MachineLearningModelInstance.generate(name=name, file=file, ml_learning_id=model.generated_from)

        model.instance = mi.id
        model.save()

        return json_response(model.to_dict())


class MachineLearningModelView(BaseView):
    serializer_class = MachineLearningModelSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ('name')

    def create(self, request, *args, **kwargs):
        property_list = ["ml_training", "name", ]

        name = get_json_r(request, "name", allow_none=False)
        ml_training = get_json_r(request, "ml_training", allow_none=False, default=None)
        request_type = RequestDataType.JSON

        user_defined_property = get_user_defined_property(request, property_list, request_type=request_type)

        return json_response(MachineLearningModel.generate(name, ml_training, **user_defined_property).to_dict())


class MachineLearningEvaluationView(BaseView):
    serializer_class = MachineLearningEvaluationSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ()

    def create(self, request, *args, **kwargs):
        property_list = ["name", "measures"]
        name = get_json_r(request, 'name', allow_none=False)
        measures = get_json_r(request, 'measures', allow_none=False, allow_type=list)

        user_defined_property = get_user_defined_property(request=request, property_list=property_list)

        mle = MachineLearningEvaluation.generate(name=name, measures=measures, **user_defined_property)

        return json_response(mle.to_dict())


class MachineLearningEvaluationExecutionView(BaseView):
    serializer_class = MachineLearningEvaluationExecutionSerializer
    cls = serializer_class.Meta.model
    queryset = cls.objects.all()
    my_filter_fields = ()

    def create(self, request, *args, **kwargs):
        property_list = ["model", "eval","eval_result"]
        model = get_json_r(request, 'model', allow_none=False)
        eval = get_json_r(request, 'eval', allow_none=False)
        eval_result = get_json_r(request, 'eval_result', allow_none=False)

        user_defined_property = get_user_defined_property(request=request, property_list=property_list)

        model_instance = MachineLearningModel.get(model)
        eval_instance:MachineLearningEvaluation = MachineLearningEvaluation.get(eval)

        mler = MachineLearningEvaluationResult.generate(name=eval,result=eval_result,eval=eval_instance)
        mlee = MachineLearningEvaluationExecution.generate(eval_name=eval, model_name=model, model=model_instance,
                                                          mle=eval_instance,result=mler, **user_defined_property)

        return json_response(mlee.to_dict())
