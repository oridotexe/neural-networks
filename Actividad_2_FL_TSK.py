# Realizado por Oriana Moreno
# NOTE: Esta actividad se planteo para peces de mar

from simpful import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Crear sistema difuso
FS = FuzzySystem(show_banner=False)

# Entradas
FS.add_linguistic_variable("peces", LinguisticVariable([
    FuzzySet(function=Triangular_MF(0, 0, 10), term="pocos"),
    FuzzySet(function=Triangular_MF(5, 15, 25), term="moderados"),
    FuzzySet(function=Triangular_MF(20, 30, 30), term="muchos")
], universe_of_discourse=[0, 30]))

FS.add_linguistic_variable("temperatura", LinguisticVariable([
    FuzzySet(function=Triangular_MF(10, 10, 18), term="baja"),
    FuzzySet(function=Triangular_MF(16, 22, 28), term="media"),
    FuzzySet(function=Triangular_MF(26, 35, 35), term="alta")
], universe_of_discourse=[10, 35]))

# Outputs constantes
FS.set_crisp_output_value("muy_bajo", 2)
FS.set_crisp_output_value("bajo", 4)
FS.set_crisp_output_value("medio", 6)
FS.set_crisp_output_value("alto", 8)
FS.set_crisp_output_value("muy_alto", 10)

# Reglas Takagi-Sugeno 
FS.add_rules([
    "IF (peces IS pocos) AND (temperatura IS baja) THEN (oxigeno IS muy_bajo)",
    "IF (peces IS pocos) AND (temperatura IS media) THEN (oxigeno IS bajo)",
    "IF (peces IS pocos) AND (temperatura IS alta) THEN (oxigeno IS medio)",
    "IF (peces IS moderados) AND (temperatura IS baja) THEN (oxigeno IS bajo)",
    "IF (peces IS moderados) AND (temperatura IS media) THEN (oxigeno IS medio)",
    "IF (peces IS moderados) AND (temperatura IS alta) THEN (oxigeno IS alto)",
    "IF (peces IS muchos) AND (temperatura IS baja) THEN (oxigeno IS medio)",
    "IF (peces IS muchos) AND (temperatura IS media) THEN (oxigeno IS alto)",
    "IF (peces IS muchos) OR (temperatura IS alta) THEN (oxigeno IS muy_alto)"
])


FS.set_variable("peces", 22)
FS.set_variable("temperatura", 30)
resultado = FS.inference()
print("\n------------------Metodo de Sugeno------------------")
print(f"Oxígeno recomendado: {resultado['oxigeno']:.2f}")
print("------------------------------------------------------")

# Superficie 3D
peces_vals = np.linspace(0, 30, 22)
temp_vals = np.linspace(10, 35, 22)
x, y = np.meshgrid(peces_vals, temp_vals)
z = np.zeros_like(x)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        FS.set_variable("peces", x[i, j])
        FS.set_variable("temperatura", y[i, j])
        z[i, j] = FS.inference()["oxigeno"]

# Gráfica 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k', alpha=0.85)
ax.set_xlabel("Cantidad de peces")
ax.set_ylabel("Temperatura (°C)")
ax.set_zlabel("Oxígeno recomendado (mg/L)")
ax.set_title("Superficie Sugeno: Oxígeno en Pecera")
plt.tight_layout()
plt.show()