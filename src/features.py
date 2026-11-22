import numpy as np
import pandas as pd
from dateutil.easter import easter

# lags candidatos (en horas). solo se usan los que son >= horizonte,
# porque al momento de predecir no se conoce nada mas reciente que eso
LAGS = [1, 2, 24, 48, 168, 336, 504, 672]


def feriados_uruguay(anios):
    no_laborables, laborables, turismo_carnaval = set(), set(), set()
    for a in anios:
        for m, d in [(1, 1), (5, 1), (7, 18), (8, 25), (12, 25)]:
            no_laborables.add(pd.Timestamp(a, m, d))
        # los laborables a veces se corren al lunes, aca no lo tengo en cuenta
        for m, d in [(1, 6), (4, 19), (5, 18), (6, 19), (10, 12), (11, 2)]:
            laborables.add(pd.Timestamp(a, m, d))

        pascua = pd.Timestamp(easter(a))
        # carnaval: lunes y martes, 48 y 47 dias antes de pascua
        turismo_carnaval.add(pascua - pd.Timedelta(days=48))
        turismo_carnaval.add(pascua - pd.Timedelta(days=47))
        # semana de turismo: de lunes a viernes antes de pascua
        for k in range(2, 7):
            turismo_carnaval.add(pascua - pd.Timedelta(days=k))
    return no_laborables, laborables, turismo_carnaval


def variables_calendario(indice):
    cal = pd.DataFrame(index=indice)
    cal["hora"] = indice.hour
    cal["dia_semana"] = indice.day_of_week
    cal["hora_semana"] = cal["dia_semana"] * 24 + cal["hora"]
    cal["mes"] = indice.month

    no_lab, lab, turismo = feriados_uruguay(range(indice.year.min(), indice.year.max() + 1))
    cerrados = list(no_lab | turismo)
    dias = indice.normalize()
    cal["feriado"] = dias.isin(cerrados).astype(int)
    cal["feriado_laborable"] = dias.isin(list(lab)).astype(int)
    # el dia despues de un feriado se acumulan llamadas
    ayer_feriado = (dias - pd.Timedelta(days=1)).isin(cerrados)
    cal["post_feriado"] = (ayer_feriado & (cal["feriado"] == 0)).astype(int)

    # dias desde el inicio de la serie, para que el modelo pueda ver la tendencia
    cal["t"] = (indice - indice[0]).days
    return cal


def armar_features(serie, horizonte=168):
    """Pasa la serie horaria a una tabla: una fila por hora, features del pasado y el target."""
    y = serie["llamadas"].astype(float)
    X = variables_calendario(serie.index)

    for k in LAGS:
        if k >= horizonte:
            X[f"lag_{k}"] = y.shift(k)

    # promedio de la misma hora en las ultimas 4 semanas (suaviza el ruido de una sola semana)
    semanas = [k for k in (168, 336, 504, 672) if k >= horizonte]
    X["media_misma_hora_4sem"] = sum(y.shift(k) for k in semanas) / len(semanas)

    # nivel general de la ultima semana que se conoce
    X["nivel_semana"] = y.shift(horizonte).rolling(168).mean()

    X["llamadas"] = y
    return X


def armar_features_futuro(serie, horizonte=168):
    # agrego las horas que quiero predecir con el target vacio y armo las features igual que siempre
    ultimo = serie.index.max()
    futuro = pd.date_range(ultimo + pd.Timedelta(hours=1), periods=horizonte, freq="h")
    extendida = pd.concat([serie[["llamadas"]], pd.DataFrame({"llamadas": np.nan}, index=futuro)])
    X = armar_features(extendida, horizonte)
    return X.loc[futuro].drop(columns="llamadas")


def columnas_features(tabla):
    return [c for c in tabla.columns if c != "llamadas"]
