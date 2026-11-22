import numpy as np
import pandas as pd


def mae(real, pred):
    return np.mean(np.abs(np.asarray(real) - np.asarray(pred)))


def rmse(real, pred):
    return np.sqrt(np.mean((np.asarray(real) - np.asarray(pred)) ** 2))


def wape(real, pred):
    # error total sobre llamadas totales. a diferencia del MAPE no explota cuando hay horas con 0
    real, pred = np.asarray(real), np.asarray(pred)
    return np.sum(np.abs(real - pred)) / np.sum(np.abs(real))


def pinball(real, pred, q):
    dif = np.asarray(real) - np.asarray(pred)
    return np.mean(np.maximum(q * dif, (q - 1) * dif))


def cobertura(real, pred):
    # que porcentaje de las veces el valor real quedo por debajo de lo predicho
    return np.mean(np.asarray(real) <= np.asarray(pred))


def horas_de_atencion(tabla, minimo=0.5):
    # horas de la semana en las que la central normalmente recibe llamadas.
    # las horas cerradas no las evaluo: predecir 0 a las 3 de la mañana es trivial y achica el error
    medias = tabla.groupby("hora_semana")["llamadas"].mean()
    return set(medias[medias > minimo].index)


def tabla_metricas(resultados, nombres):
    base = mae(resultados["real"], resultados["Naive estacional"])
    filas = []
    for m in nombres:
        e = mae(resultados["real"], resultados[m])
        filas.append({
            "Modelo": m,
            "MAE": e,
            "RMSE": rmse(resultados["real"], resultados[m]),
            "WAPE (%)": 100 * wape(resultados["real"], resultados[m]),
            "Mejora vs naive (%)": 100 * (1 - e / base),
        })
    return pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)
