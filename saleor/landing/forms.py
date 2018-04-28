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

    def clean(self):
        form_data = self.cleaned_data

        if not form_data.get("desde") or not form_data.get("hasta"):
            raise forms.ValidationError(
                "Ingresar ubicacion de salida y de llegada")

        if form_data["desde"] == form_data["hasta"]:
            raise forms.ValidationError(
                "Los lugares deben ser diferentes")

        entradas = [
            form_data["t_entrada_lunes"],
            form_data["t_entrada_martes"],
            form_data["t_entrada_miercoles"],
            form_data["t_entrada_jueves"],
            form_data["t_entrada_viernes"]]

        salidas = [
            form_data["t_salida_lunes"],
            form_data["t_salida_martes"],
            form_data["t_salida_miercoles"],
            form_data["t_salida_jueves"],
            form_data["t_salida_viernes"]]

        tiempos = zip(entradas, salidas)
        tiempos = [ts for ts in tiempos if all(ts)]
        if not tiempos:
            raise forms.ValidationError(
                "Favor ingresar almenos un horario")
        else:
            for ts in tiempos:
                if ts[0] > 24 or ts[1] > 24:
                    raise forms.ValidationError(
                        "Hora debe ser menor a 24")
                elif ts[0] < 0 or ts[1] < 0:
                    raise forms.ValidationError(
                        "Hora debe ser menor a 24")
                elif ts[0] >= ts[1]: \
                    raise forms.ValidationError(
                        "t debe ser menor a t salida")

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
            "preferencia_1",
            "preferencia_2",
            "preferencia_3",
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
            'preferencia_1': forms.RadioSelect(attrs={'display': 'inline-block'}),
            'preferencia_2': forms.RadioSelect(attrs={'display': 'inline-block'}),
            'preferencia_3': forms.RadioSelect(attrs={'display': 'inline-block'}),
            'sugerencias': forms.Textarea(attrs={'placeholder': 'Tu opinion es muy valiosa'}),
        }
