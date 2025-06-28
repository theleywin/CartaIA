import random
from typing import List
from schemas.bdi import Belief
from utils.evaluador_fuzzy import evaluar_calidad_difusa, calcular_metricas_action_plan

def generar_individuo(action_plan, size=5) -> List[str]:
    individuo = random.sample(action_plan, size)
    return individuo

def mutar(action_plan, individuo: List[str], tasa=0.4, debug=False) -> List[str]:
    nuevo = individuo.copy()
    for i in range(len(nuevo)):
        if random.random() < tasa:
            original = nuevo[i]
            nuevo[i] = random.choice(action_plan)
            if debug:
                print(f"[MUTAR] Sustituido '{original}' por '{nuevo[i]}' en posición {i}")
    if debug and nuevo != individuo:
        print(f"[MUTAR] Resultado final: {nuevo}")
    return nuevo

def cruzar(padre1: List[str], padre2: List[str], debug=False) -> List[str]:
    punto = random.randint(1, len(padre1) - 2)
    hijo = padre1[:punto] + padre2[punto:]
    if debug:
        print(f"[CRUZAR] Punto de cruce: {punto}")
        print(f"[CRUZAR] Padre1: {padre1}")
        print(f"[CRUZAR] Padre2: {padre2}")
        print(f"[CRUZAR] Hijo:   {hijo}")
    return hijo

def calcular_fitness(plan: List[str], estado_estudiante: Belief, debug=False) -> float:
    metricas = calcular_metricas_action_plan(plan, estado_estudiante)
    fitness = evaluar_calidad_difusa(
        metricas["cobertura"],
        metricas["progresion_val"],
        metricas["variedad_val"],
        metricas["alineacion_val"]
    )
    if debug:
        print(f"[FITNESS] Plan: {plan}")
        print(f"[FITNESS] Métricas: {metricas}")
        print(f"[FITNESS] Valor fitness: {fitness:.4f}")
    return fitness

async def optimizar_plan(
    action_plan: List[str],
    estado_estudiante: Belief,
    generaciones=30,
    tamano_poblacion=20,
    debug=False
) -> List[str]:

    poblacion = [generar_individuo(action_plan) for _ in range(tamano_poblacion)]

    for gen in range(generaciones):
        puntuados = sorted(
            [(plan, calcular_fitness(plan, estado_estudiante, debug=debug)) for plan in poblacion],
            key=lambda x: x[1],
            reverse=True
        )

        mejor_fit = puntuados[0][1]
        mejor_plan = puntuados[0][0]
        avg_fit = sum(score for _, score in puntuados) / len(puntuados)

        if debug:
            print(f"[GEN {gen+1}] Mejor plan: {mejor_plan}")

        elite = [p for p, _ in puntuados[:5]]

        nueva_poblacion = elite.copy()
        while len(nueva_poblacion) < tamano_poblacion:
            padre1, padre2 = random.sample(elite, 2)
            hijo = mutar(action_plan, cruzar(padre1, padre2, debug=debug), debug=debug)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    final_plan = max(poblacion, key=lambda p: calcular_fitness(p, estado_estudiante))
    return final_plan