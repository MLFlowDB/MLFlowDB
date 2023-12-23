from django.urls import path, include


from rest_framework_mongoengine.routers import DefaultRouter

from apps.crud.data_gathering.view import OriginDataView, OriginDataCollectionView

data_gathering_router = DefaultRouter()

data_gathering_router.register(r'origin_data', OriginDataView,
                            basename="origin_data")
data_gathering_router.register(r'origin_data_collection', OriginDataCollectionView,
                            basename="origin_data_collection")

urlpatterns = [
        path('origin_data_group_create/',
         view = OriginDataView.as_view({'post': 'group_create'})),
]

urlpatterns += data_gathering_router.urls