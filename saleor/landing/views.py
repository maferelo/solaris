import logging

from django.template.response import TemplateResponse
from django.shortcuts import redirect

from .forms import EncuestaForm
from .utils import get_routes
from .models import Encuesta, Routes

logger = logging.getLogger(__name__)


def landing_home(request):
    return TemplateResponse(
        request, 'landing/landing_home.html',
        {'parent': None})


def landing_encuesta(request):
    logger.info(request.POST)
    form = EncuestaForm(request.POST or None)
    if form.is_valid():
        logger.info(form.cleaned_data)
        form.save()
        return redirect('landing_urls:landing_home')
    return TemplateResponse(
        request, 'landing/landing_encuesta.html',
        {'parent': None,
         'form': form})


def landing_solver(request):
    customers_q = Encuesta.objects.all()
    logger.info(len(customers_q))
    routes = get_routes(customers_q)
    return TemplateResponse(
        request, 'landing/landing_solver.html',
        {'parent': None,
         'routes': routes})


{'t_salida_viernes': None, 't_salida_martes': None, 't_entrada_lunes': 7, 't_entrada_jueves': None, 't_entrada_miercoles': None, 'correo': None, 'preferencia_2': 'BU', 't_salida_miercoles': None, 't_entrada_martes': None, 'preferencia_1': 'BU', 't_entrada_viernes': None, 'hasta': '(-75.59977873751222 6.222686105062644)', 'sugerencias': '', 'edad': 21, 'genero': 'M', 'modo': 'VE', 'gasto': 300000, 't_salida_lunes': 10, 't_salida_jueves': None, 'preferencia_3': 'BU', 'desde': '(-75.6069426 6.2169782)', 'tiempo_trayecto': 50}



