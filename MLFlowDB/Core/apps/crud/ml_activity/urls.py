from django.urls import path, include

from apps.crud.ml_activity.view import MachineLearningImplementationView, UserDefinedMeasureView, \
    UserDefinedAlgorithmClassView, MachineLearningImplementationSourceCodeView, MachineLearningTaskView, \
    MachineLearningTrainingView, MachineLearningModelView, MachineLearningModelInstanceView, \
    MachineLearningEvaluationView, MachineLearningEvaluationExecutionView

from rest_framework_mongoengine.routers import DefaultRouter

ml_activity_router = DefaultRouter()

ml_activity_router.register(r'machine_learning_implementation', MachineLearningImplementationView,
                            basename="machine_learning_implementation")
ml_activity_router.register(r'user_defined_measure', UserDefinedMeasureView,
                            basename="user_defined_measure")
ml_activity_router.register(r'user_defined_algorithm_class', UserDefinedAlgorithmClassView,
                            basename="user_defined_algorithm_class")
ml_activity_router.register(r'machine_learning_training', MachineLearningTrainingView,
                            basename="machine_learning_training")
ml_activity_router.register(r'machine_learning_task', MachineLearningTaskView,
                            basename="machine_learning_task")
ml_activity_router.register(r'machine_learning_model', MachineLearningModelView,
                            basename="machine_learning_model")
ml_activity_router.register(r'machine_learning_model_instance', MachineLearningModelInstanceView,
                            basename="machine_learning_model_instance")
ml_activity_router.register(r'machine_learning_evaluation', MachineLearningEvaluationView,
                            basename="machine_learning_evaluation")
ml_activity_router.register(r'machine_learning_evaluation_execution', MachineLearningEvaluationExecutionView,
                            basename="machine_learning_evaluation_execution")
urlpatterns = [
    # path('machine_learning_implementation', view=MachineLearningImplementationView.as_view({'post': 'add'})),
    path('machine_learning_implementation_source_code/',
         view=MachineLearningImplementationSourceCodeView.as_view({'post': 'add'})),
]

urlpatterns += ml_activity_router.urls
