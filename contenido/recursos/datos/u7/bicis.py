"""Datos de las bicicletas preparados como en la unidad 2."""
import numpy as np
import pandas as pd

h = pd.read_csv("hour.csv", parse_dates=["dteday"])
X = pd.DataFrame({
    "temp_c": h.temp * 41, "humedad": h.hum * 100,
    "viento_kmh": h.windspeed * 67,
    "tiempo": h.weathersit.clip(upper=3),
    "hora_sen": np.sin(2 * np.pi * h.hr / 24),
    "hora_cos": np.cos(2 * np.pi * h.hr / 24),
    "hora": h.hr, "laborable": h.workingday, "festivo": h.holiday,
    "dia_semana": h.weekday, "mes": h.mnth, "año": h.yr})
X.loc[h.dteday == "2011-03-10", "humedad"] = np.nan
X["humedad"] = X.humedad.interpolate()
y = h.cnt
ent = (h.dteday < "2012-07-01").to_numpy()
pru = ~ent
