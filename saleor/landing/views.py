from django.template.response import TemplateResponse
from django.shortcuts import get_object_or_404, redirect

from .forms import CityCreateForm, EncuestaForm


def landing_home(request):
    return TemplateResponse(
        request, 'landing/landing_home.html',
        {'parent': None})


def landing_encuesta(request):
    form = EncuestaForm(request.POST or None)
    if form.is_valid():
        return redirect('landing_urls:landing_home')
    return TemplateResponse(
        request, 'landing/landing_encuesta.html',
        {'parent': None,
         'form': form})
