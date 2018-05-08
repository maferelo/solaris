from django import forms
from mapwidgets.widgets import GooglePointFieldWidget

from .models import City, Encuesta, HORAS


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

        try:
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
        except KeyError:
            raise forms.ValidationError(
                "Favor solo indicar la hora, ejemplo 10. No 10:15"
            )

        tiempos = zip(entradas, salidas)
        tiempos = [ts for ts in tiempos if isinstance(ts[0], int) and isinstance(ts[1], int)]
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
                elif ts[0] >= ts[1]:
                    raise forms.ValidationError(
                        "Hora de entrada debe ser menor a la hora de salida")

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
            'edad': forms.Select(),
            'genero': forms.Select(),
            'vehiculo': forms.Select(),
            'tiempo_trayecto': forms.Select(),
            'gasto': forms.Select(),
            'desde': GooglePointFieldWidget,
            'hasta': GooglePointFieldWidget,
            't_entrada_lunes': forms.Select(),
            't_salida_lunes': forms.Select(),
            't_entrada_martes': forms.Select(),
            't_salida_martes': forms.Select(),
            't_entrada_miercoles': forms.Select(),
            't_salida_miercoles': forms.Select(),
            't_entrada_jueves': forms.Select(),
            't_salida_jueves': forms.Select(),
            't_entrada_viernes': forms.Select(),
            't_salida_viernes': forms.Select(),
            'preferencia_1': forms.RadioSelect(),
            'preferencia_2': forms.RadioSelect(),
            'preferencia_3': forms.RadioSelect(),
            'sugerencias': forms.Textarea(attrs={'placeholder': 'Tu opinion es muy valiosa'}),
        }
