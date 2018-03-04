from django.conf.urls import url

from . import views


urlpatterns = [
    url(r'^encuesta/$', views.landing_encuesta, name='landing_encuesta'),
    url(r'^$', views.landing_home, name='landing_home')
]
