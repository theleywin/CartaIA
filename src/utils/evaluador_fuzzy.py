import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List
from schemas.bdi import Belief

# --- Variables lingüísticas
cobertura = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cobertura')
progresion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'progresion')
variedad = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'variedad')
alineacion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'alineacion')
calidad = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'calidad')

# --- Funciones de membresía
cobertura['baja'] = fuzz.trimf(cobertura.universe, [0.0, 0.0, 0.3])
cobertura['media'] = fuzz.trimf(cobertura.universe, [0.2, 0.5, 0.8])
cobertura['alta'] = fuzz.trimf(cobertura.universe, [0.6, 1.0, 1.0])

progresion['lenta'] = fuzz.trimf(progresion.universe, [0.0, 0.0, 0.3])
progresion['moderada'] = fuzz.trimf(progresion.universe, [0.2, 0.5, 0.8])
progresion['rápida'] = fuzz.trimf(progresion.universe, [0.6, 1.0, 1.0])

variedad['baja'] = fuzz.trimf(variedad.universe, [0.0, 0.0, 0.3])
variedad['media'] = fuzz.trimf(variedad.universe, [0.2, 0.5, 0.8])
variedad['alta'] = fuzz.trimf(variedad.universe, [0.6, 1.0, 1.0])

alineacion['pobre'] = fuzz.trimf(alineacion.universe, [0.0, 0.0, 0.3])
alineacion['moderada'] = fuzz.trimf(alineacion.universe, [0.2, 0.5, 0.8])
alineacion['fuerte'] = fuzz.trimf(alineacion.universe, [0.6, 1.0, 1.0])

calidad['baja'] = fuzz.trimf(calidad.universe, [0.0, 0.0, 0.3])
calidad['media'] = fuzz.trimf(calidad.universe, [0.2, 0.5, 0.8])
calidad['alta'] = fuzz.trimf(calidad.universe, [0.6, 1.0, 1.0])


# --- Reglas difusas
reglas = [
    ctrl.Rule(cobertura['alta'] & progresion['moderada'] & variedad['alta'] & alineacion['fuerte'], calidad['alta']),
    ctrl.Rule(cobertura['media'] & progresion['moderada'] & variedad['media'] & alineacion['moderada'], calidad['media']),
    ctrl.Rule(cobertura['baja'] & progresion['lenta'] & variedad['baja'] & alineacion['pobre'], calidad['baja']),
    ctrl.Rule(cobertura['alta'] & alineacion['fuerte'], calidad['alta']),
    ctrl.Rule(variedad['alta'] & progresion['moderada'], calidad['media']),
    ctrl.Rule(cobertura['media'] & progresion['rápida'] & variedad['alta'] & alineacion['fuerte'], calidad['alta']),
    ctrl.Rule(cobertura['alta'] & progresion['moderada'] & alineacion['moderada'], calidad['alta']),
    ctrl.Rule(cobertura['alta'] & variedad['media'] & alineacion['moderada'], calidad['media']),
    ctrl.Rule(cobertura['media'] & progresion['moderada'] & alineacion['fuerte'], calidad['media']),
    ctrl.Rule(cobertura['media'] & progresion['lenta'] & variedad['media'] & alineacion['moderada'], calidad['media']),
    ctrl.Rule(cobertura['baja'] & progresion['moderada'] & variedad['alta'], calidad['media']),
    ctrl.Rule(variedad['media'] & alineacion['fuerte'], calidad['media']),
    ctrl.Rule(cobertura['media'] & alineacion['moderada'], calidad['media']),
    ctrl.Rule(progresion['lenta'] & variedad['baja'] & alineacion['pobre'], calidad['baja']),
    ctrl.Rule(cobertura['baja'] & progresion['lenta'] & alineacion['moderada'], calidad['baja']),
    ctrl.Rule(cobertura['media'] & variedad['baja'] & alineacion['pobre'], calidad['baja']),
    ctrl.Rule(progresion['lenta'] & alineacion['moderada'], calidad['baja']),
]

sistema_ctrl = ctrl.ControlSystem(reglas)
sistema_simulador = ctrl.ControlSystemSimulation(sistema_ctrl)


def evaluar_calidad_difusa(
    cobertura_val: float,
    progresion_val: float,
    variedad_val: float,
    alineacion_val: float
) -> float:
    sistema_simulador.input['cobertura'] = cobertura_val
    sistema_simulador.input['progresion'] = progresion_val
    sistema_simulador.input['variedad'] = variedad_val
    sistema_simulador.input['alineacion'] = alineacion_val

    try:
        sistema_simulador.compute()
        return round(sistema_simulador.output['calidad'], 4)
    except KeyError:
        return 0.0


def calcular_metricas_action_plan(plan: List[str], estado: Belief) -> dict:
    total_errores = len(estado.misconceptions)
    total_preferencias = len(estado.learning_preferences)

    # Métrica 1: cobertura
    cobertura_val = sum(
        1 for e in estado.misconceptions if any(e.lower() in paso.lower() for paso in plan)
    )
    cobertura_val = cobertura_val / total_errores if total_errores else 0.0

    # Métrica 2: progresión (por dificultad de los pasos) Esto es muy mejorable, se puede usar un modelo de lenguaje, pero
    # las api-keys gratuitas no permiten hacer esto de forma eficiente, así que se hace de forma muy simple
    dificultad = 0
    for paso in plan:
        paso_l = paso.lower()
        if "visualizar" in paso_l or "presentarán" in paso_l or "presentar" in paso_l or "verán" in paso_l or "ver" in paso_l:
            dificultad += 1
        elif "ejercicio práctico" in paso_l or "problema práctico" in paso_l:
            dificultad += 0.9
        elif "ejemplo avanzado" in paso_l:
            dificultad += 0.8
        elif "ejemplo" in paso_l:
            dificultad += 0.6
        elif "practicar" in paso_l or "introducir" in paso_l:
            dificultad += 0.4
        elif "teoría" in paso_l or "teórico" in paso_l or "explicación" in paso_l or "resolverán" in paso_l or "resolver" in paso_l or "explicarán" in paso_l or "explicar" in paso_l:
            dificultad += 0.3
        elif "implementar" in paso_l or "implementación" in paso_l or "implementación de código" in paso_l or "implementación de un algoritmo" in paso_l:
            dificultad += 0.3
        elif "revisar" in paso_l or "revisión" in paso_l or "revisar" in paso_l or "revisar código" in paso_l or "revisión de código" in paso_l:
            dificultad += 0.3
        elif "analizar" in paso_l or "análisis" in paso_l or "analizar" in paso_l or "análisis de código" in paso_l :
            dificultad += 0.3
        else:
            dificultad += 0.5
    # Normalizamos la dificultad entre 0 y 1
    progresion_val = min(dificultad / len(plan), 1.0) if plan else 0.0
    # Métrica 3: variedad de pasos
    variedad_val = len(set(plan)) / len(plan) if plan else 0.0

    # Métrica 4: alineación con preferencias de aprendizaje
    alineacion_val = sum(
        1 for pref in estado.learning_preferences if any(pref.lower() in paso.lower() for paso in plan)
    )
    alineacion_val = alineacion_val / total_preferencias if total_preferencias else 0.0

    return {
        "cobertura": round(cobertura_val, 3),
        "progresion_val": round(progresion_val, 3),
        "variedad_val": round(variedad_val, 3),
        "alineacion_val": round(alineacion_val, 3),
    }