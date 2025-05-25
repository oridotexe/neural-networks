# Realizado por Oriana Moreno
# NOTE: Esta actividad se planteo para peces de mar

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Inputs
peces = ctrl.Antecedent(np.arange(0, 31, 1), 'peces')                   # peces = [0, 30]
temperatura = ctrl.Antecedent(np.arange(10, 36, 1), 'temperatura')      # temperatura = [10, 35]

# Outputs
oxigeno = ctrl.Consequent(np.arange(0, 11, 1), 'oxigeno')               # oxigeno = [0, 10]

# Funciones de membresía
peces['pocos'] = fuzz.trimf(peces.universe, [0, 0, 10])
peces['moderados'] = fuzz.trimf(peces.universe, [5, 15, 25])
peces['muchos'] = fuzz.trimf(peces.universe, [20, 30, 30])

temperatura['baja'] = fuzz.trimf(temperatura.universe, [10, 10, 18])
temperatura['media'] = fuzz.trimf(temperatura.universe, [16, 22, 28])
temperatura['alta'] = fuzz.trimf(temperatura.universe, [26, 35, 35])

oxigeno['bajo'] = fuzz.trimf(oxigeno.universe, [0, 0, 3])
oxigeno['medio'] = fuzz.trimf(oxigeno.universe, [2, 5, 8])
oxigeno['alto'] = fuzz.trimf(oxigeno.universe, [7, 10, 10])

# Reglas
rule1 = ctrl.Rule(peces['pocos'] & temperatura['baja'], oxigeno['bajo'])
rule2 = ctrl.Rule(peces['pocos'] & temperatura['media'], oxigeno['medio'])
rule3 = ctrl.Rule(peces['pocos'] & temperatura['alta'], oxigeno['alto'])
rule4 = ctrl.Rule(peces['moderados'] & temperatura['baja'], oxigeno['medio'])
rule5 = ctrl.Rule(peces['moderados'] & temperatura['media'], oxigeno['medio'])
rule6 = ctrl.Rule(peces['moderados'] & temperatura['alta'], oxigeno['alto'])
rule7 = ctrl.Rule(peces['muchos'] & temperatura['baja'], oxigeno['medio'])
rule8 = ctrl.Rule(peces['muchos'] & temperatura['media'], oxigeno['alto'])
rule9 = ctrl.Rule(peces['muchos'] | temperatura['alta'], oxigeno['alto'])

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]

# Sistema y simulador
sistema_oxigeno = ctrl.ControlSystem(rules)
simulador = ctrl.ControlSystemSimulation(sistema_oxigeno)

# Ejemplo: 22 peces y 30°C
simulador.input['peces'] = 22
simulador.input['temperatura'] = 30
simulador.compute()

# Resultado
print("\n------------------Metodo de Mandani------------------")
print(f"Oxígeno recomendado: {simulador.output['oxigeno']:.2f} ")
print("-------------------------------------------------------")
oxigeno.view(sim=simulador)


# Datos 3D
peces_vals = np.arange(0, 31, 5)
temp_vals = np.arange(10, 36, 5)
X, Y = np.meshgrid(peces_vals, temp_vals)
Z = np.zeros_like(X, dtype=float)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        simulador.input['peces'] = X[i, j]
        simulador.input['temperatura'] = Y[i, j]
        simulador.compute()
        Z[i, j] = simulador.output['oxigeno']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Cantidad de peces')
ax.set_ylabel('Temperatura (°C)')
ax.set_zlabel('Oxígeno recomendado (mg/L)')
ax.set_title('Controlador Difuso - Mamdani')
plt.tight_layout()
plt.show()