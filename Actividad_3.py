# Realizado por Oriana Moreno CI 29929240

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

class Perceptron:
    def __init__(self, n_classes, n_features, lr=0.01, epochs=1000):
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.W = np.random.randn(n_classes, n_features + 1) * 0.01  
        self.errors_ = []
        self.acc_ = []

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        scores = np.dot(X_bias, self.W.T)
        return np.argmax(scores, axis=1)

    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X_bias, y):
                scores = np.dot(self.W, xi)
                predicted = np.argmax(scores)
                if predicted != target:
                    self.W[target] += self.lr * xi
                    self.W[predicted] -= self.lr * xi
                    errors += 1
            self.errors_.append(errors)
            acc = np.mean(self.predict(X) == y)
            self.acc_.append(acc)

class Adaline:
    def __init__(self, n_classes, n_features, lr=0.01, epochs=1000):
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.W = np.random.randn(n_classes, n_features + 1) * 0.01
        self.costs_ = []
        self.acc_ = []

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        net_input = np.dot(X_bias, self.W.T)
        return np.argmax(net_input, axis=1)

    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        Y = np.eye(self.n_classes)[y]  

        for _ in range(self.epochs):
            net_input = np.dot(X_bias, self.W.T)
            errors = Y - net_input
            self.W += self.lr * np.dot(errors.T, X_bias) / X.shape[0]
            cost = (errors ** 2).sum() / 2.0
            self.costs_.append(cost)
            acc = np.mean(self.predict(X) == y)
            self.acc_.append(acc)

# --- Cargar datos ---
try:
    data = np.loadtxt('glass.data', delimiter=',')
except FileNotFoundError:
    print("Error: No se encontró el archivo glass.data")
    sys.exit()

X = data[:, 1:-1]
y = data[:, -1].astype(int) 

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
n_classes = y.max() + 1 


# --- Entrenar modelos ---
perc = Perceptron(n_classes=n_classes, n_features=X.shape[1], lr=0.01, epochs=100)
perc.fit(X_train, y_train)

adal = Adaline(n_classes=n_classes, n_features=X.shape[1], lr=0.01, epochs=100)
adal.fit(X_train, y_train)

# --- Evaluar ---
y_pred_perc = perc.predict(X_test)
y_pred_adal = adal.predict(X_test)

acc_perc = np.mean(y_pred_perc == y_test) * 100
acc_adal = np.mean(y_pred_adal == y_test) * 100

print(f"\n--- Resultados del Perceptron ---")
print(f"Precisión en test: {acc_perc:.2f}%")

print(f"\n--- Resultados de Adaline ---")
print(f"Precisión en test: {acc_adal:.2f}%")

# --- Graficar ---
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(perc.errors_, label="Errores - Perceptrón", color="blue")
plt.xlabel("Épocas")
plt.ylabel("Errores")
plt.title("Errores por época - Perceptrón")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(adal.costs_, label="Coste - Adaline", color="red")
plt.xlabel("Épocas")
plt.ylabel("Costo")
plt.title("Costo por época - Adaline")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- Grafico de precisión ---
plt.figure(figsize=(10,4))
plt.plot(perc.acc_, label="Perceptrón", color="blue")
plt.plot(adal.acc_, label="Adaline", color="red")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.title("Precisión por época")
plt.grid(True)
plt.legend()
plt.show()

# --- Grafico de datos ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
labels = np.unique(y)

plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1],
                label=f'Clase {label}', alpha=0.7, edgecolors='k', s=50)
labels = np.unique(y)  

plt.legend()
plt.tight_layout()
plt.show()