from django.urls import path, include

urlpatterns = [
    path('data_gathering/',include(('apps.crud.data_gathering.urls', 'apps.crud'), namespace='data_gathering')),
    path('data_transforming/',include(('apps.crud.data_transforming.urls', 'apps.crud'), namespace='data_transforming')),
    path('ml_activity/',include(('apps.crud.ml_activity.urls', 'apps.crud'), namespace='ml_activity'))
]
