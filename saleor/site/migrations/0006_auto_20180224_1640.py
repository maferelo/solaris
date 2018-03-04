# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2018-02-24 22:40
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('site', '0005_auto_20170906_0556'),
    ]

    operations = [
        migrations.AlterField(
            model_name='authorizationkey',
            name='name',
            field=models.CharField(choices=[('facebook', 'Facebook-Oauth2'), ('google-oauth2', 'Google-Oauth2')], max_length=20, verbose_name='name'),
        ),
    ]
