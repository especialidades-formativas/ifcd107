"""Preparación y métricas comunes del caso práctico."""
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from banco import num, cat

def prep_lineal():                        # escalar y one-hot
    return make_column_transformer(
        (StandardScaler(), num),
        (OneHotEncoder(handle_unknown="ignore"), cat))

def mejora(y, p, frac=0.1):               # lift del mejor 10 %
    n = int(frac * len(y))
    top = np.argsort(-np.asarray(p))[:n]
    return np.asarray(y)[top].mean() / np.mean(y)

def metricas(y, p):
    return (f"AUC {roc_auc_score(y, p):.3f}  AP"
            f" {average_precision_score(y, p):.3f}"
            f"  lift@10% {mejora(y, p):.2f}")

from sklearn.compose import make_column_transformer as _mct
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

def potenciacion(**kw):                   # el modelo de referencia
    ordinal = _mct(("passthrough", num),
                   (OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), cat))
    k = len(num)                          # las categóricas van detrás
    opciones = dict(categorical_features=list(range(k, k + len(cat))),
                    learning_rate=0.05, max_iter=300, random_state=0)
    opciones.update(kw)
    return make_pipeline(ordinal,
                         HistGradientBoostingClassifier(**opciones))

num_todas, cat_todas = list(num), list(cat)   # para quitar variables
