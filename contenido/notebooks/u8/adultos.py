"""Censo de EE. UU. de 1994 (Adult): ¿ingresos > 50 000 $?"""
import pandas as pd
from sklearn.model_selection import train_test_split

cols = ["edad", "empleo", "peso", "estudios", "años_estudio",
        "estado_civil", "ocupacion", "relacion", "raza", "sexo",
        "ganancias", "perdidas", "horas", "pais", "ingresos"]
d = pd.read_csv("adult.data", names=cols, skipinitialspace=True,
                na_values="?")
d["alto"] = (d.ingresos == ">50K").astype(int)
X = d.drop(columns=["ingresos", "alto", "peso", "estudios"])
X_e, X_p, y_e, y_p = train_test_split(X, d.alto, test_size=0.3,
                                      stratify=d.alto, random_state=0)
