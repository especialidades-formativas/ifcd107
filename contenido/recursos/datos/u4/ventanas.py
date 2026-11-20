"""Ventanas de 24 horas de las bicicletas para predecir la hora 25."""
import numpy as np
import pandas as pd

h = pd.read_csv("hour.csv", parse_dates=["dteday"])
t = h.dteday + pd.to_timedelta(h.hr, unit="h")
serie = np.log1p(h.cnt.to_numpy()).astype("float32")
exo = np.column_stack([      # lo que se sabe de antemano
    np.sin(2 * np.pi * h.hr / 24), np.cos(2 * np.pi * h.hr / 24),
    h.workingday, h.temp]).astype("float32")
L = 24
idx = np.arange(L, len(h))
seguidas = (t.to_numpy()[idx] - t.to_numpy()[idx - L]
            == np.timedelta64(L, "h"))     # sin horas perdidas
idx = idx[seguidas]
X = np.stack([np.column_stack([serie[i - L:i], exo[i - L + 1:i + 1]])
              for i in idx])               # (ventanas, 24, 5)
y = serie[idx]
ent = (h.dteday.to_numpy()[idx] < np.datetime64("2012-07-01"))
pru = ~ent
real = h.cnt.to_numpy()[idx]
