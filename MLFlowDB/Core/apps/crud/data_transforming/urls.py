from django.urls import path, include

from apps.crud.data_transforming.view import DataPreparationComponentView, DataPreparationComponentFrameworkView, \
    DataPreparationPipelineView, DataPreparationPipelineExecutionView, DatasetView

from rest_framework_mongoengine.routers import DefaultRouter

data_transforming_router = DefaultRouter()

data_transforming_router.register(r'dataset', DatasetView,
                            basename="dataset")

urlpatterns = [
    path('data_preparation/', view=DataPreparationComponentView.as_view({'post': 'add'})),
    path('data_preparation_framework/', view=DataPreparationComponentFrameworkView.as_view({'post': 'add'})),
    path('data_preparation_pipeline/', view=DataPreparationPipelineView.as_view({'post': 'add'})),
    path('data_preparation_pipeline_execution/', view=DataPreparationPipelineExecutionView.as_view({'post': 'add'})),
]

urlpatterns += data_transforming_router.urls