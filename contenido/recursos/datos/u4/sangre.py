"""BloodMNIST: imágenes de 28x28 en color, escaladas a [0, 1]."""
import numpy as np

d = np.load("bloodmnist.npz")
X_e, y_e = d["train_images"] / 255.0, d["train_labels"].ravel()
X_v, y_v = d["val_images"] / 255.0, d["val_labels"].ravel()
X_p, y_p = d["test_images"] / 255.0, d["test_labels"].ravel()
nombres = ["basófilo", "eosinófilo", "eritroblasto",
           "granulocito inmaduro", "linfocito", "monocito",
           "neutrófilo", "plaqueta"]
