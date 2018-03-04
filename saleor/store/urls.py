from django.conf.urls import url

from . import views


urlpatterns = [
    url(r'^store/$', views.store_details, name='store_details'),
    url(r'^create/$', views.create, name='store_create'),
]
