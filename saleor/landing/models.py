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
    (False, 'No')
)


class City(models.Model):
    name = models.CharField(max_length=255)
    coordinates = models.PointField(help_text="To generate the map for your location")
    city_hall = models.PointField(blank=True, null=True)

    def __unicode__(self):
        return self.name


class Encuesta(models.Model):
    MUJER = 'M'
    HOMBRE = 'H'
    GENDER_CHOICES = (
        (MUJER, 'Mujer'),
        (HOMBRE, 'Hombre')
    )

    MODO_CHOICES = (
        ('VE', 'Vehiculo particular'),
        ('BU', 'Bus'),
        ('MO', 'Moto'),
        ('BI', 'Bicicleta'),
        ('CA', 'Caminando'),
    )

    edad = models.IntegerField(
        'edad',
        validators=[MinValueValidator(4),
                    MaxValueValidator(100)])
    genero = models.CharField(
        'sexo',
        max_length=1,
        choices=GENDER_CHOICES,
        default=MUJER
    )
    modo = models.CharField(
        "Modo de transporte",
        max_length=2,
        choices=MODO_CHOICES,
        default='VE'
    )
    tiempo_trayecto = models.IntegerField(
        'tiempo aprox. trayecto (minutos)',
        validators=[MinValueValidator(4),
                    MaxValueValidator(240)])
    gasto = models.IntegerField(
        'gasto mensual en transporte',
        validators=[MinValueValidator(0),
                    MaxValueValidator(10000000)])

    desde = models.PointField(
        "Ubicación de salida",
        help_text="De donde sales")
    hasta = models.PointField(
        "Ubicación de entrada",
        help_text="A donde llegas")

    t_entrada_lunes = models.IntegerField(
        'Hora de entrada - Lunes (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_salida_lunes = models.IntegerField(
        'Hora de entrada - Lunes (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_entrada_martes = models.IntegerField(
        'Hora de entrada - Martes (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_salida_martes = models.IntegerField(
        'Hora de entrada - Martes (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_entrada_miercoles = models.IntegerField(
        'Hora de entrada - Miércoles (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_salida_miercoles = models.IntegerField(
        'Hora de entrada - Miércoles (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_entrada_jueves = models.IntegerField(
        'Hora de entrada - Jueves (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_salida_jueves = models.IntegerField(
        'Hora de entrada - Jueves (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_entrada_viernes = models.IntegerField(
        'Hora de entrada - Viernes (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])
    t_salida_viernes = models.IntegerField(
        'Hora de entrada - Viernes (opcional)',
        blank=True,
        validators=[MinValueValidator(1),
                    MaxValueValidator(24)])

    preferencia = models.BooleanField(
        "¿Utilizarias este producto?",
        choices=TRUE_FALSE_CHOICES, default=True)

    correo = models.EmailField(
        'email (opcional)', blank=True, unique=True)
    sugerencias = models.TextField(
        'sugerencias (opcional)', blank=True)

    def __unicode__(self):
        return self.salida

