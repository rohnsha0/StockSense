from django.urls import path
from . import views

urlpatterns = [
    path('', views.root, name='root'),
    path('query/<str:symbol>/', views.query, name='query'),
    path('query/v2/<str:symbol>/', views.query_v2, name='query_v2'),
    path('query/v3/<str:symbol>/', views.query_v3, name='query_v3'),
    path('info/<str:symbol>/', views.info, name='info'),
    path('ltp/<str:symbol>/', views.ltp, name='ltp'),
    path('technical/<str:symbol>/', views.technicals, name='technicals'),
    path('prediction/<str:symbol>/', views.prediction, name='prediction'),
    path('prediction/hr/<str:symbol>/', views.hourly_prediction, name='hourly_prediction'),
]