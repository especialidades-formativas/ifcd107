"""SMS: texto y etiqueta (1 = spam), sin repetidos, 75/25."""
import pandas as pd
from sklearn.model_selection import train_test_split

d = pd.read_csv("sms2.tsv", sep="\t", names=["clase", "texto"],
                quoting=3)
d["spam"] = (d.clase == "spam").astype(int)
u = d.drop_duplicates("texto")          # sin mensajes repetidos
t_e, t_p, y_e, y_p = train_test_split(
    u.texto, u.spam, test_size=0.25, stratify=u.spam, random_state=0)
