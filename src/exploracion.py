import numpy as np


def autocorrelacion(y, max_lag=400):
    """Autocorrelacion muestral para los lags 0 a max_lag (en el 0 es 1 por definicion)."""
    # es la correlacion de pearson entre la serie y ella misma corrida k horas.
    # la calculo a mano porque son cuatro lineas y asi me queda claro que es lo que estoy graficando
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    var = np.sum(y ** 2)
    acf = [1.0]
    for k in range(1, max_lag + 1):
        # cada valor por el de hace k horas: si la serie se parece a si misma corrida k,
        # los productos quedan grandes y positivos. dividir por la varianza lo deja entre -1 y 1
        acf.append(np.sum(y[k:] * y[:-k]) / var)
    return np.array(acf)


def mapa_calor(serie):
    # promedio de llamadas por dia de la semana y hora: una grilla de 7x24 que resume
    # la "semana tipica". es la forma mas rapida de ver la doble estacionalidad de un vistazo
    tmp = serie[["llamadas"]].copy()
    tmp["dia"] = tmp.index.day_of_week
    tmp["hora"] = tmp.index.hour
    return tmp.pivot_table(index="dia", columns="hora", values="llamadas", aggfunc="mean")
