"""Bank Marketing: llamadas de un banco portugués, 2008-2010."""
import pandas as pd
from sklearn.model_selection import train_test_split

d = pd.read_csv("bank-additional-full.csv", sep=";")
meses = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
         "nov", "dec"]
orden = d.month.map({m: i for i, m in enumerate(meses)})
d["año"] = 2008 + (orden.diff() < 0).cumsum()   # filas por fecha
d["y"] = (d.y == "yes").astype(int)
X = d.drop(columns=["y", "duration", "año"])    # sin la duración
y = d.y
hist = (d.año <= 2009).to_numpy()               # 2008 y 2009
X_e, X_p, y_e, y_p = train_test_split(
    X[hist], y[hist], test_size=0.25, stratify=y[hist],
    random_state=0)
X_f, y_f = X[~hist], y[~hist]                   # 2010: el futuro
num = ["age", "campaign", "pdays", "previous", "emp.var.rate",
       "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
cat = ["job", "marital", "education", "default", "housing", "loan",
       "contact", "month", "day_of_week", "poutcome"]
