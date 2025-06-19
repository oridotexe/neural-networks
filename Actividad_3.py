import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score

# 1. Cargar glass.data
data = np.loadtxt('glass.data', delimiter=',')

# 2. Filtrar solo clases 1 y 2 (columna -1 es la clase)
X = data[:, 1:-1]  # columnas de atributos
y = data[:, -1]    # columna de la clase

mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]

# Convertir y a etiquetas binarias: clase 1 -> -1, clase 2 -> +1
y = np.where(y == 1, -1, 1)

# 3. Seleccionar SOLO 2 features para visualización (RI y Mg)
# Columna 0 (RI) y columna 2 (Mg)
X = X[:, [0, 2]]

# 4. Normalizar datos
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# 5. Perceptron
class Perceptron:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# 6.  Adaline
class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = net_input
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

# 7. Función para graficar frontera de decisión
def plot_decision_regions(X, y, classifier, title):
    resolution = 0.02

    markers = ('s', 'x')
    colors = ('red', 'blue')
    cmap = plt.cm.RdBu

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Clase {cl}')

    plt.xlabel('RI (normalizado)')
    plt.ylabel('Mg (normalizado)')
    plt.title(title)
    plt.legend()

# 8. Entrenar modelos
ppn = Perceptron(eta=0.01, n_iter=10)
ppn.fit(X, y)

ada = AdalineGD(eta=0.01, n_iter=50)
ada.fit(X, y)

# 10. Graficar
plt.figure(figsize=(12,5))

# Perceptron
plt.subplot(1,2,1)
plot_decision_regions(X, y, classifier=ppn, title='Perceptrón - Frontera de decisión')

# Adaline
plt.subplot(1,2,2)
plot_decision_regions(X, y, classifier=ada, title='Adaline - Frontera de decisión')

plt.tight_layout()
plt.show()

# 11. Resultados
y_pred_ppn = ppn.predict(X)
y_pred_ada = ada.predict(X)

acc_ppn = accuracy_score(y, y_pred_ppn) * 100
acc_ada = accuracy_score(y, y_pred_ada) * 100

print("\n--- Resultados del Perceptrón ---")
print("Pesos finales:", ppn.w_)
print(f"Precisión: {acc_ppn:.2f}%")

print("\n--- Resultados del Adaline ---")
print("Pesos finales:", ada.w_)
print(f"Precisión: {acc_ada:.2f}%")