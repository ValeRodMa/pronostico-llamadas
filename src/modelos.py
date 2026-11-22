import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NaiveEstacional:
    """Baseline: predice lo mismo que paso a la misma hora de la ultima semana conocida."""

    # con horizonte de 1 semana eso es lag_168; con horizontes mas largos la semana
    # pasada todavia no se conoce, asi que repite el lag mas chico que este permitido
    def __init__(self, columna="lag_168"):
        self.columna = columna

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.columna].to_numpy()


class HoltWinters:
    """Suavizado exponencial con nivel, tendencia y estacionalidad semanal.

    Es el modelo clasico de referencia. No usa la tabla de features: trabaja directo
    sobre la serie, asi que lo envuelvo con la misma interfaz fit/predict que los demas
    para poder meterlo en el mismo backtest. Ojo: predict() asume que las filas a
    predecir vienen justo despues del final del entrenamiento, que es exactamente lo
    que pasa en el walk-forward y al pronosticar la semana siguiente.
    """

    def fit(self, X, y):
        # estacionalidad aditiva y no multiplicativa: de noche hay horas con 0 llamadas
        # y la multiplicativa necesita valores positivos.
        # con periodo 168 alcanza, porque el patron semanal ya contiene al diario
        with warnings.catch_warnings():
            # el optimizador avisa que corta antes de converger del todo; lo probe
            # con 5000 iteraciones y la prediccion da exactamente igual
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.ajustado_ = ExponentialSmoothing(
                np.asarray(y, dtype=float), trend="add", seasonal="add",
                seasonal_periods=168,
            ).fit()
        return self

    def predict(self, X):
        return self.ajustado_.forecast(len(X))


def ridge(columnas, alpha=1.0):
    # para un modelo lineal la hora de la semana tiene que ir en one-hot,
    # si no estaria asumiendo que las llamadas crecen linealmente con la hora
    categoricas = ["hora_semana", "mes"]
    numericas = [c for c in columnas if c not in categoricas]
    prepro = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", StandardScaler(), numericas),
    ])
    return make_pipeline(prepro, Ridge(alpha=alpha))


def gradient_boosting(**kwargs):
    # early_stopping apagado: por defecto separa una validacion al azar, que en series no corresponde
    params = dict(max_iter=400, learning_rate=0.05, early_stopping=False, random_state=0)
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def crear_modelos(columnas):
    # el lag mas chico de la tabla marca que semana es "la ultima conocida" para el naive
    lag_naive = min(int(c.split("_")[1]) for c in columnas if c.startswith("lag_"))
    return {
        "Naive estacional": NaiveEstacional(f"lag_{lag_naive}"),
        "Holt-Winters": HoltWinters(),
        "Ridge": ridge(columnas),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.5,
                                               n_jobs=-1, random_state=0),
        "Gradient Boosting": gradient_boosting(),
    }


def modelo_cuantil(q):
    # misma idea pero minimizando la pinball loss: predice el cuantil q en vez de la media.
    # lo dejo mas regularizado que el otro: sin esto el cuantil da bien en entrenamiento
    # pero queda demasiado angosto con datos nuevos (cubre bastante menos que el q pedido)
    return gradient_boosting(
        loss="quantile", quantile=q, max_iter=200, min_samples_leaf=60, max_leaf_nodes=15)
