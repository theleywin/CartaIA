import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def get_execution_times(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    tiempos = [item["execution_time"] for item in json_data]
    return tiempos

def media_intervalo_confianza(tiempos, confianza=0.95, n_bootstrap=10000):
    bootstrap_means = []
    n = len(tiempos)
    
    for _ in range(n_bootstrap):
        muestra = np.random.choice(tiempos, size=n, replace=True)
        bootstrap_means.append(np.mean(muestra))
    
    media = np.mean(bootstrap_means)
    percentil_inf = (1 - confianza) / 2 * 100
    percentil_sup = (1 + confianza) / 2 * 100
    lim_inf, lim_sup = np.percentile(bootstrap_means, [percentil_inf, percentil_sup])
    
    return {
        "media_poblacional_estimada": media,
        "intervalo_confianza": (round(lim_inf,2), round(lim_sup,2)),
        "nivel_confianza": confianza,
        "metodo": "Bootstrapping (no paramétrico)"
    }