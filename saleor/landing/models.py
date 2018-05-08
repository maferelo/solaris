from __future__ import unicode_literals

import datetime
from decimal import Decimal

from django.core.validators import (MinValueValidator,
                                    MaxValueValidator, DecimalValidator)
from django.contrib.gis.db import models
from django_prices.models import Price, PriceField
from prices import PriceRange


TRUE_FALSE_CHOICES = (
    (True, 'Si'),
    (False, 'No'))

HORAS = [
    (x, "{}:00 a.m.".format(x)) if x < 12 else (x, "{}:00 p.m.".format(x)) for x in range(24 + 1)]

EDADES = (
    (12, '12-14'),
    (15, '15-24'),
    (25, '25-44'),
    (45, '45-64'),
    (65, '65+'))

TIEMPOS = (
    (0, '0-14'),
    (15, '15-29'),
    (30, '30-44'),
    (45, '45-59'),
    (61, '60+'))

GASTOS = (
    (0, '$0-$100.000'),
    (100000, '$100.000-$200.000'),
    (200000, '$200.000-$300.000'),
    (300000, '$300.000+'))


class City(models.Model):
    name = models.CharField(max_length=255)
    coordinates = models.PointField(help_text="To generate the map for your location")
    city_hall = models.PointField(blank=True, null=True)

    def __unicode__(self):
        return self.name


class Encuesta(models.Model):

    GENDER_CHOICES = (
        ('M', 'Mujer'),
        ('H', 'Hombre'))

    MODO_CHOICES = (
        ('VE', 'Vehiculo particular'),
        ('BU', 'Bus/Metro'),
        ('MO', 'Moto'),
        ('BI', 'Bicicleta'),
        ('CA', 'Caminando'))

    PREFERENCIAS_1_CHOICES = (
        ('BU', 'Bus: $2100 - 50min'),
        ('OM', 'Omibus: $3000 - 50min'),
        ('PA', 'Automovil particular: $7000 - 30min'))

    PREFERENCIAS_2_CHOICES = (
        ('BU', 'Bus: $2100 - 50min'),
        ('OM', 'Omibus: $3500 - 40min'),
        ('PA', 'Automovil particular: $7000 - 30min'))

    PREFERENCIAS_3_CHOICES = (
        ('BU', 'Bus: $2100 - 50min'),
        ('OM', 'Omibus: $4000 - 30min'),
        ('PA', 'Automovil particular: $7000 - 30min'))

    TEXTO_AYUDA = """Recuerda que para utilizar tu ubicación debes tener el GPS prendido,
          de lo contrario puedes escribirla en el cuadro de texto que dice
          <i>Ingresar el nombre o tu dirección acá</i>, elegir la mejor opción y
          despues colocar el marcador"""

    edad = models.IntegerField(
        'edad',
        choices=EDADES)
    genero = models.CharField(
        'sexo',
        max_length=1,
        choices=GENDER_CHOICES)
    modo = models.CharField(
        "Modo de transporte",
        max_length=2,
        choices=MODO_CHOICES)
    tiempo_trayecto = models.IntegerField(
        'tiempo aprox. trayecto (minutos)',
        choices=TIEMPOS)
    gasto = models.IntegerField(
        'gasto mensual en transporte',
        choices=GASTOS)

    desde = models.PointField(
        "Ubicación de alojamiento - e.j. Casa",
        help_text=TEXTO_AYUDA)
    hasta = models.PointField(
        "Ubicación de trabajo, estudio, colegio u otros",
        help_text="")

    t_entrada_lunes = models.IntegerField(
        'Lunes - Hora ingreso (opcional)',
        blank=True, null=True, choices=HORAS)
    t_salida_lunes = models.IntegerField(
        'Lunes - Hora fin de la actividad (opcional)',
        blank=True, null=True, choices=HORAS)
    t_entrada_martes = models.IntegerField(
        'Martes - Hora ingreso (opcional)',
        blank=True, null=True, choices=HORAS)
    t_salida_martes = models.IntegerField(
        'Martes - Hora fin de la actividad (opcional)',
        blank=True, null=True, choices=HORAS)
    t_entrada_miercoles = models.IntegerField(
        'Miércoles - Hora ingreso (opcional)',
        blank=True, null=True, choices=HORAS)
    t_salida_miercoles = models.IntegerField(
        'Miércoles - Hora fin de la actividad (opcional)',
        blank=True, null=True, choices=HORAS)
    t_entrada_jueves = models.IntegerField(
        'Jueves - Hora de ingreso (opcional)',
        blank=True, null=True, choices=HORAS)
    t_salida_jueves = models.IntegerField(
        'Jueves - Hora fin de la actividad (opcional)',
        blank=True, null=True, choices=HORAS)
    t_entrada_viernes = models.IntegerField(
        'Viernes - Hora de ingreso (opcional)',
        blank=True, null=True, choices=HORAS)
    t_salida_viernes = models.IntegerField(
        'Viernes - Hora fin de la actividad (opcional)',
        blank=True, null=True, choices=HORAS)

    preferencia_1 = models.CharField(
        "¿Cual opción consideras mas conveniente? *Valores típicos de un trayecto al centro",
        max_length=2,
        choices=PREFERENCIAS_1_CHOICES)

    preferencia_2 = models.CharField(
        "¿Cual opción consideras mas conveniente?",
        max_length=2,
        choices=PREFERENCIAS_2_CHOICES)

    preferencia_3 = models.CharField(
        "¿Cual opción consideras mas conveniente?",
        max_length=2,
        choices=PREFERENCIAS_3_CHOICES)

    correo = models.EmailField(
        'email (opcional)', blank=True, null=True)
    sugerencias = models.TextField(
        'sugerencias (opcional)', blank=True)

    def __unicode__(self):
        return self.salida


class Routes(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    assigment = models.FileField(upload_to='assigments/')

