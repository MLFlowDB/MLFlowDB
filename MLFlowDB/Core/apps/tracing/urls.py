from django.urls import path, include

from apps.tracing.view import TracingView

urlpatterns = [
    # path('machine_learning_implementation', view=MachineLearningImplementationView.as_view({'post': 'add'})),
    path('trace/',
         view=TracingView.as_view({'post': 'trace'})),
    path('trace_result/<str:id>/',
         view=TracingView.as_view({'get': 'get_value'})),
]