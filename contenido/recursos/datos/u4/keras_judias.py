"""Judías preparadas para Keras: escaladas y con clases 0..6."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

j = pd.read_csv("judias.csv").drop_duplicates()
clases = sorted(j.Class.unique())
X = j.drop(columns="Class").to_numpy("float32")
y = j.Class.map(clases.index).to_numpy()
X_e, X_p, y_e, y_p = train_test_split(X, y, test_size=0.25,
                                      stratify=y, random_state=0)
media, desv = X_e.mean(0), X_e.std(0)
X_e, X_p = (X_e - media) / desv, (X_p - media) / desv
