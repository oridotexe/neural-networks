# Realizado por Oriana Moreno
# NOTE: Esta actividad se planteo para peces de mar

import numpy as np
import matplotlib.pyplot as plt
from Actividad_2_FL_Mamdani import mamdani
from Actividad_2_FL_Takagi import takagi_sugeno


simulador_mamdani, oxigeno_mamdani = mamdani()
FS_sugeno = takagi_sugeno()

peces_vals = np.arange(0, 31, 1) 
# Valor variable de temperatura 
input_temperatura = 20 
oxigeno_mamdani_vals = []
oxigeno_sugeno_vals = []

for peces_value in peces_vals:
    # Mamdani
    simulador_mamdani.input['peces'] = peces_value
    simulador_mamdani.input['temperatura'] = input_temperatura
    simulador_mamdani.compute()
    oxigeno_mamdani_vals.append(simulador_mamdani.output['oxigeno'])
    
    # Takagi-Sugeno
    FS_sugeno.set_variable("peces", peces_value)
    FS_sugeno.set_variable("temperatura", input_temperatura)
    oxigeno_sugeno_vals.append(FS_sugeno.inference()["oxigeno"])

# ---- GRAFICA ---- #
plt.figure(figsize=(10, 6))
plt.plot(peces_vals, oxigeno_mamdani_vals, label='Mamdani', color='blue')
plt.plot(peces_vals, oxigeno_sugeno_vals, label='Takagi-Sugeno', color='green')
plt.xlabel('Cantidad de peces')
plt.ylabel('Oxígeno recomendado (mg/L)')
plt.title('Comparación Mamdani vs Takagi-Sugeno')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()