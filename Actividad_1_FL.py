# Oriana V. Moreno R.  29.929.240

import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

folder = "funciones resultantes"
if not os.path.exists(folder):
    os.makedirs(folder)

# Definir los universos de discurso
temp_universe = np.arange(25, 70.1, 0.1)   # Temperatura [25, 70]
vel_universe = np.arange(1.2, 3.21, 0.01)  # Velocidad [1.2, 3.2]
act_universe = np.arange(0, 100.1, 0.1)    # Actividad [0, 100]
vent_universe = np.arange(0, 1.01, 0.01)   # Ventilador [0, 1]

# Conjuntos
# Temperatura
temp_T1 = fuzz.trapmf(temp_universe, [4.75, 22.75, 27.25, 45.25])
temp_T2 = fuzz.trapmf(temp_universe, [27.25, 45.25, 49.75, 67.75])
temp_T3 = fuzz.trapmf(temp_universe, [49.75, 67.75, 72.25, 90.25])

# Velocidad
vel_V1 = fuzz.trapmf(vel_universe, [0.3, 1.1, 1.3, 2.1])
vel_V2 = fuzz.trapmf(vel_universe, [1.3, 2.1, 2.3, 3.1])
vel_V3 = fuzz.trapmf(vel_universe, [2.3, 3.1, 3.3, 4.1])

# Actividad
act_A1 = fuzz.trapmf(act_universe, [-45, -5, 5, 45])
act_A2 = fuzz.trapmf(act_universe, [5, 45, 55, 95])
act_A3 = fuzz.trapmf(act_universe, [55, 95, 105, 145])

# Output
vent_OFF = fuzz.trapmf(vent_universe, [-0.9, -0.1, 0.1, 0.9])
vent_ON = fuzz.trapmf(vent_universe, [0.1, 0.9, 1.1, 1.9])

# Visualizar los conjuntos difusos
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# Temperatura
ax[0, 0].plot(temp_universe, temp_T1, 'b', linewidth=1.5, label='T1')
ax[0, 0].plot(temp_universe, temp_T2, 'g', linewidth=1.5, label='T2')
ax[0, 0].plot(temp_universe, temp_T3, 'r', linewidth=1.5, label='T3')
ax[0, 0].set_title('Temperatura')
ax[0, 0].legend()

# Velocidad
ax[0, 1].plot(vel_universe, vel_V1, 'b', linewidth=1.5, label='V1')
ax[0, 1].plot(vel_universe, vel_V2, 'g', linewidth=1.5, label='V2')
ax[0, 1].plot(vel_universe, vel_V3, 'r', linewidth=1.5, label='V3')
ax[0, 1].set_title('Velocidad')
ax[0, 1].legend()

# Actividad
ax[1, 0].plot(act_universe, act_A1, 'b', linewidth=1.5, label='A1')
ax[1, 0].plot(act_universe, act_A2, 'g', linewidth=1.5, label='A2')
ax[1, 0].plot(act_universe, act_A3, 'r', linewidth=1.5, label='A3')
ax[1, 0].set_title('Actividad')
ax[1, 0].legend()

# Ventilador
ax[1, 1].plot(vent_universe, vent_OFF, 'b', linewidth=1.5, label='OFF')
ax[1, 1].plot(vent_universe, vent_ON, 'r', linewidth=1.5, label='ON')
ax[1, 1].set_title('Ventilador')
ax[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(folder, 'conjuntos_difusos.png'))
plt.close()

# Inputs
temperatura = 50
velocidad = 2.0
actividad = 63

print("Inputs: ")
print(f"Temperatura: {temperatura}")
print(f"Velocidad: {velocidad}")
print(f"Actividad: {actividad}")

# PASO 1: Fuzzificacion
# Temperatura
temp_grado_T1 = fuzz.interp_membership(temp_universe, temp_T1, temperatura)
temp_grado_T2 = fuzz.interp_membership(temp_universe, temp_T2, temperatura)
temp_grado_T3 = fuzz.interp_membership(temp_universe, temp_T3, temperatura)

# Velocidad
vel_grado_V1 = fuzz.interp_membership(vel_universe, vel_V1, velocidad)
vel_grado_V2 = fuzz.interp_membership(vel_universe, vel_V2, velocidad)
vel_grado_V3 = fuzz.interp_membership(vel_universe, vel_V3, velocidad)

# Actividad
act_grado_A1 = fuzz.interp_membership(act_universe, act_A1, actividad)
act_grado_A2 = fuzz.interp_membership(act_universe, act_A2, actividad)
act_grado_A3 = fuzz.interp_membership(act_universe, act_A3, actividad)

# PASO 2: Reglas difusas - Inferencia
# Regla 1: IF T1 AND V1 AND A1 THEN OFF
rule1 = np.fmin(np.fmin(temp_grado_T1, vel_grado_V1), act_grado_A1)
# Regla 2: IF T3 OR V3 OR A3 THEN ON
rule2 = np.fmax(np.fmax(temp_grado_T3, vel_grado_V3), act_grado_A3)
# Regla 3: IF NOT T1 AND NOT V1 AND NOT A2 THEN NOT OFF
rule3 = np.fmin(np.fmin(1 - temp_grado_T1, 1 - vel_grado_V1), 1 - act_grado_A2)
# Regla 4: IF NOT T2 AND NOT V1 AND NOT A1 THEN NOT ON 
rule4 = np.fmin(np.fmin(1 - temp_grado_T2, 1 - vel_grado_V1), 1 - act_grado_A1)
# Regla 5: IF NOT T1 AND NOT V2 AND NOT A1 THEN NOT OFF
rule5 =  np.fmin(np.fmin(1 - temp_grado_T1, 1 - vel_grado_V2), 1 - act_grado_A1)
# Reglas 6: IF T2 AND  V2 AND  A2 THEN ON
rule6 = np.fmin(np.fmin(temp_grado_T2, vel_grado_V2), act_grado_A2)

# PASO 3: Composicion
implicacion_off = np.fmin(rule1, vent_OFF)

implicacion_on = np.fmax(
    np.fmax(
        np.fmin(rule2, vent_ON),  
        np.fmin(rule3, vent_ON)   
    ),
    np.fmax(
        np.fmin(rule4, vent_ON),  
        np.fmin(rule5, vent_ON)   
    )
)
implicacion_on = np.fmax(implicacion_on, np.fmin(rule6, vent_ON))

# Agregación
agregado = np.fmax(implicacion_off, implicacion_on)

# PASO 4: Desfuzzification
if np.any(agregado):
    defuzz_centroid = fuzz.defuzz(vent_universe, agregado, 'centroid')
    defuzz_result = fuzz.interp_membership(vent_universe, agregado, defuzz_centroid)
else:
    defuzz_centroid = 0
    defuzz_result = 0


print("\nResultados de Defuzificación:")
print(f"Crisp: {defuzz_centroid:.4f}")
print(f"Salida: {'ON' if defuzz_centroid > 0.5 else 'OFF'}")

# Visualizacion 
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='ON')
plt.fill_between(vent_universe, implicacion_off, alpha=0.3, color='blue', label='Implicación OFF')
plt.fill_between(vent_universe, implicacion_on, alpha=0.3, color='red', label='Implicación ON')
plt.fill_between(vent_universe, agregado, alpha=0.7, color='gray', label='Agregación')
plt.axvline(x=defuzz_centroid, color='green', linestyle='-', linewidth=2, label=f'Centroide: {defuzz_centroid:.4f}')
plt.title('Defuzificación por Centroide')
plt.ylabel('Grado de Pertenencia')
plt.xlabel('Ventilador')
plt.legend()
plt.savefig(os.path.join(folder, 'defuzificacion.png'))
plt.close()

########################################
##             Parte 2                ##
########################################

# Paso 2: Reglas - Inferencia Difusa
# Modificacion de la regla 6
# IF T2 AND V2 AND A2 THEN OFF
rule6_mod = np.fmin(np.fmin(temp_grado_T2, vel_grado_V2), act_grado_A2)

# PASO 3: Composicion
implicacion_off_mod = np.fmax(
    np.fmin(rule1, vent_OFF),
    np.fmin(rule6, vent_OFF)  
)

implicacion_on_mod = np.fmax(
    np.fmax(
        np.fmin(rule2, vent_ON),  
        np.fmin(rule3, vent_ON)   
    ),
    np.fmax(
        np.fmin(rule4, vent_ON),  
        np.fmin(rule5, vent_ON)   
    )
)

# Agregacion 
agregado_mod = np.fmax(implicacion_off_mod, implicacion_on_mod)

# PASO 4: desfuzzificacion 
if np.any(agregado_mod):
    defuzz_centroid_mod = fuzz.defuzz(vent_universe, agregado_mod, 'centroid')
    defuzz_result_mod = fuzz.interp_membership(vent_universe, agregado_mod, defuzz_centroid_mod)
else:
    defuzz_centroid_mod = 0
    defuzz_result_mod = 0


print("\nResultados de Defuzificacion (con Regla 6 modificada):")
print(f"Crisp: {defuzz_centroid_mod:.4f}")
print(f"Salida: {'ON' if defuzz_centroid_mod > 0.5 else 'OFF'}")

# Visualizacion
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='ON')
plt.fill_between(vent_universe, implicacion_off_mod, alpha=0.3, color='blue', label='Implicación OFF (mod)')
plt.fill_between(vent_universe, implicacion_on_mod, alpha=0.3, color='red', label='Implicación ON (mod)')
plt.fill_between(vent_universe, agregado_mod, alpha=0.7, color='grey', label='Agregación (mod)')
plt.axvline(x=defuzz_centroid_mod, color='green', linestyle='-', linewidth=2, label=f'Centroide: {defuzz_centroid_mod:.4f}')
plt.title('Defuzificación por Centroide (Regla 6 modificada)')
plt.ylabel('Grado de Pertenencia')
plt.xlabel('Ventilador')
plt.legend()
plt.savefig(os.path.join(folder, 'defuzificacion_modificada.png'))
plt.close()
