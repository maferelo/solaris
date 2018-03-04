from django import forms
from mapwidgets.widgets import GooglePointFieldWidget

from .models import City, Encuesta


class CityCreateForm(forms.ModelForm):

    class Meta:
        model = City
        fields = ("name", "coordinates", "city_hall")
        widgets = {
            'coordinates': GooglePointFieldWidget,
            'city_hall': GooglePointFieldWidget,
        }


class EncuestaForm(forms.ModelForm):

    class Meta:
        model = Encuesta
        fields = (
            "edad",
            "genero",
            "modo",
            "tiempo_trayecto",
            "gasto",
            "desde",
            "hasta",
            "t_entrada_lunes",
            "t_salida_lunes",
            "t_entrada_martes",
            "t_salida_martes",
            "t_entrada_miercoles",
            "t_salida_miercoles",
            "t_entrada_jueves",
            "t_salida_jueves",
            "t_entrada_viernes",
            "t_salida_viernes",
            "preferencia",
            "correo",
            "sugerencias")
        widgets = {
            'edad': forms.TextInput(attrs={'placeholder': 'ej. 21'}),
            'genero': forms.RadioSelect(),
            'vehiculo': forms.RadioSelect(),
            'tiempo_trayecto': forms.TextInput(attrs={'placeholder': 'ej. 50'}),
            'gasto': forms.TextInput(attrs={'placeholder': 'ej. 300000'}),
            'desde': GooglePointFieldWidget,
            'hasta': GooglePointFieldWidget,
            't_entrada_lunes': forms.TextInput(attrs={'placeholder': 'ej. 7'}),
            't_salida_lunes': forms.TextInput(attrs={'placeholder': 'ej. 17'}),
            't_entrada_martes': forms.TextInput(attrs={'placeholder': 'ej. 7'}),
            't_salida_martes': forms.TextInput(attrs={'placeholder': 'ej. 17'}),
            't_entrada_miercoles': forms.TextInput(attrs={'placeholder': 'ej. 7'}),
            't_salida_miercoles': forms.TextInput(attrs={'placeholder': 'ej. 17'}),
            't_entrada_jueves': forms.TextInput(attrs={'placeholder': 'ej. 7'}),
            't_salida_jueves': forms.TextInput(attrs={'placeholder': 'ej. 17'}),
            't_entrada_viernes': forms.TextInput(attrs={'placeholder': 'ej. 7'}),
            't_salida_viernes': forms.TextInput(attrs={'placeholder': 'ej. 17'}),
            'preferencia': forms.RadioSelect(),
            'sugerencias': forms.Textarea(attrs={'placeholder': 'Tu opinion es muy valiosa'}),
        }
