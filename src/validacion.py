import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, TimeSeriesSplit

from src.features import columnas_features
from src.metricas import mae


def particiones(n_filas, semanas=8, horizonte=168):
    # walk-forward: entreno con todo lo anterior y pruebo con la semana siguiente, y voy corriendo
    tscv = TimeSeriesSplit(n_splits=semanas, test_size=horizonte)
    return list(tscv.split(np.arange(n_filas)))


def backtest(tabla, modelos, semanas=8, horizonte=168):
    cols = columnas_features(tabla)
    X, y = tabla[cols], tabla["llamadas"]
    partes = []
    for i, (ent, prueba) in enumerate(particiones(len(tabla), semanas, horizonte)):
        res = pd.DataFrame({"real": y.iloc[prueba].to_numpy(), "semana": i + 1},
                           index=y.index[prueba])
        for nombre, modelo in modelos.items():
            modelo.fit(X.iloc[ent], y.iloc[ent])
            # no tiene sentido predecir llamadas negativas
            res[nombre] = np.clip(modelo.predict(X.iloc[prueba]), 0, None)
        partes.append(res)
    return pd.concat(partes)


def comparar_validaciones(tabla, modelo, semanas_eval=6, semanas_futuro=4, horizonte=168):
    """
    Separo las ultimas semanas como 'futuro' que no se toca.
    Con el resto estimo el error de dos formas (k-fold al azar y walk-forward), midiendo las dos
    sobre las mismas semanas, y despues lo comparo con el error que el modelo tuvo de verdad en el futuro.
    """
    cols = columnas_features(tabla)
    corte = len(tabla) - semanas_futuro * horizonte
    pasado, futuro = tabla.iloc[:corte], tabla.iloc[corte:]
    X, y = pasado[cols], pasado["llamadas"]
    evaluar = np.arange(len(pasado) - semanas_eval * horizonte, len(pasado))

    # k-fold al azar: cada fila se predice con un modelo que no la vio, pero si vio sus vecinas
    # (la hora anterior, la siguiente, y tambien semanas posteriores)
    pred_kfold = np.zeros(len(pasado))
    for ent, prueba in KFold(n_splits=10, shuffle=True, random_state=0).split(X):
        m = clone(modelo).fit(X.iloc[ent], y.iloc[ent])
        pred_kfold[prueba] = m.predict(X.iloc[prueba])

    # walk-forward sobre esas mismas semanas: cada una se predice solo con lo anterior
    pred_walk = []
    for ent, prueba in particiones(len(pasado), semanas_eval, horizonte):
        m = clone(modelo).fit(X.iloc[ent], y.iloc[ent])
        pred_walk.append(m.predict(X.iloc[prueba]))

    final = clone(modelo).fit(X, y)

    return {
        "K-fold al azar": mae(y.iloc[evaluar], pred_kfold[evaluar]),
        "Walk-forward": mae(y.iloc[evaluar], np.concatenate(pred_walk)),
        "Error real en el futuro": mae(futuro["llamadas"], final.predict(futuro[cols])),
    }
