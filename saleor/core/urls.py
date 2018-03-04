from django.conf.urls import url

from . import views
from ..landing import views as landing_views


urlpatterns = [
    url(r'^$', landing_views.landing_home, name='landing_home'),
    url(r'^style-guide/', views.styleguide, name='styleguide'),
]
