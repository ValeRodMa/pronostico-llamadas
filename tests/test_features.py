import numpy as np
import pandas as pd
import pytest

from src.features import armar_features, armar_features_futuro, columnas_features


def serie_de_prueba(semanas=8, semilla=0):
    rng = np.random.default_rng(semilla)
    indice = pd.date_range("2026-01-05", periods=semanas * 168, freq="h")
    return pd.DataFrame({"llamadas": rng.poisson(20, len(indice))}, index=indice)


@pytest.mark.parametrize("horizonte", [24, 168])
def test_no_usa_datos_del_futuro(horizonte):
    # si predigo la hora t con horizonte h, solo puedo saber lo que paso hasta t - h.
    # cambio todos los valores posteriores a t - h y las features de t tienen que quedar iguales
    serie = serie_de_prueba()
    t = serie.index[6 * 168 + 30]

    modificada = serie.copy()
    modificada.loc[modificada.index > t - pd.Timedelta(hours=horizonte), "llamadas"] = 9999

    f1 = armar_features(serie, horizonte)
    f2 = armar_features(modificada, horizonte)
    cols = columnas_features(f1)
    pd.testing.assert_series_equal(f1.loc[t, cols], f2.loc[t, cols])


def test_si_cambio_el_pasado_las_features_cambian():
    # control del test anterior: si toco algo que si se puede ver, tiene que notarse
    serie = serie_de_prueba()
    t = serie.index[6 * 168 + 30]
    modificada = serie.copy()
    modificada.loc[t - pd.Timedelta(hours=168), "llamadas"] = 9999

    f1 = armar_features(serie, 168)
    f2 = armar_features(modificada, 168)
    assert f1.loc[t, "lag_168"] != f2.loc[t, "lag_168"]


def test_features_futuro_completas():
    serie = serie_de_prueba()
    futuro = armar_features_futuro(serie, 168)
    assert len(futuro) == 168
    assert futuro.index[0] == serie.index[-1] + pd.Timedelta(hours=1)
    assert not futuro.isna().any().any()
