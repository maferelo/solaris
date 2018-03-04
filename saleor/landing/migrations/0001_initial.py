# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2018-02-24 22:44
from __future__ import unicode_literals

import django.contrib.gis.db.models.fields
import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='City',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('coordinates', django.contrib.gis.db.models.fields.PointField(help_text='To generate the map for your location', srid=4326)),
                ('city_hall', django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326)),
            ],
        ),
        migrations.CreateModel(
            name='Encuesta',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('edad', models.IntegerField(validators=[django.core.validators.MinValueValidator(4), django.core.validators.MaxValueValidator(100), django.core.validators.DecimalValidator(2, 0)], verbose_name='edad')),
                ('genero', models.CharField(choices=[('M', 'Mujer'), ('H', 'Hombre')], default='M', max_length=1, verbose_name='sexo')),
                ('vehiculo', models.BooleanField(choices=[(True, 'Si'), (False, 'No')], default=True)),
                ('tiempo_trayecto', models.IntegerField(validators=[django.core.validators.MinValueValidator(4), django.core.validators.MaxValueValidator(180), django.core.validators.DecimalValidator(2, 0)], verbose_name='tiempo aprox. trayecto (minutos)')),
                ('gasto', models.IntegerField(validators=[django.core.validators.MinValueValidator(100), django.core.validators.MaxValueValidator(1000000), django.core.validators.DecimalValidator(2, 0)], verbose_name='gasto mensual en transporte')),
                ('lugar_salida', django.contrib.gis.db.models.fields.PointField(help_text='Utiliza tu ubicación o ingresa la dirección', srid=4326)),
                ('t_entrada_lunes', models.TimeField(verbose_name='Hora de entrada')),
                ('t_salida_lunes', models.TimeField(verbose_name='Hora de salida')),
                ('lugar_entrada_lunes', django.contrib.gis.db.models.fields.PointField(help_text='To generate the map for your location', srid=4326)),
                ('preferencia', models.BooleanField(choices=[(True, 'Si'), (False, 'No')], default=True)),
                ('correo', models.EmailField(blank=True, max_length=254, unique=True, verbose_name='email')),
                ('sugerencias', models.TextField(blank=True, verbose_name='sugerencias')),
            ],
        ),
    ]
