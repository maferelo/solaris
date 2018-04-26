from django.conf.urls import url

from . import views


urlpatterns = [
    url(r'^encuesta/$', views.landing_encuesta, name='landing_encuesta'),
    url(r'^optimize/$', views.landing_solver, name='landing_solver'),
    url(r'^$', views.landing_home, name='landing_home')
]
