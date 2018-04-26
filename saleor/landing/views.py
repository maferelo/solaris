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
    form = EncuestaForm(request.POST or None)
    if form.is_valid():
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


