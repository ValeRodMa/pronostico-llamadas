"""
Corre el pipeline completo desde la terminal.

    python main.py simular       genera datos de prueba en data/simulado/
    python main.py evaluar       compara los modelos con validacion walk-forward
    python main.py pronosticar   predice la semana siguiente y calcula operadores
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import datos, erlang, features, metricas, modelos, validacion

HORIZONTE = 168  # una semana


def elegir_datos(ruta):
    if ruta:
        return ruta
    if Path(datos.RUTA_REAL).exists():
        return datos.RUTA_REAL
    if not Path(datos.RUTA_SIMULADA).exists():
        print("No hay datos reales, genero datos simulados...")
        datos.guardar_simulacion()
    return datos.RUTA_SIMULADA


def evaluar(ruta, semanas):
    serie = datos.cargar_serie(ruta)
    tabla = features.armar_features(serie, HORIZONTE).dropna()
    mods = modelos.crear_modelos(features.columnas_features(tabla))

    print(f"Datos: {ruta} ({len(serie)} horas)")
    print(f"Backtest walk-forward con {semanas} semanas de prueba...\n")
    res = validacion.backtest(tabla, mods, semanas, HORIZONTE)

    abiertas = metricas.horas_de_atencion(tabla)
    res = res[tabla.loc[res.index, "hora_semana"].isin(abiertas)]
    tabla_m = metricas.tabla_metricas(res, list(mods))
    print(tabla_m.round(2).to_string(index=False))

    Path("resultados").mkdir(exist_ok=True)
    tabla_m.to_csv("resultados/metricas.csv", index=False)
    res.to_csv("resultados/predicciones_backtest.csv")
    print("\nGuardado en resultados/")


def mejor_modelo(tabla, semanas=4):
    # backtest corto solo para decidir con que modelo pronostico la media.
    # antes usaba siempre gradient boosting, pero no tiene sentido comparar modelos
    # en "evaluar" y despues pronosticar con uno fijo: el ganador depende de los datos
    # (con tendencia fuerte suele ganar ridge, porque los arboles no extrapolan)
    mods = modelos.crear_modelos(features.columnas_features(tabla))
    res = validacion.backtest(tabla, mods, semanas, HORIZONTE)
    abiertas = metricas.horas_de_atencion(tabla)
    res = res[tabla.loc[res.index, "hora_semana"].isin(abiertas)]
    return metricas.tabla_metricas(res, list(mods)).iloc[0]["Modelo"]


def pronosticar(ruta, cuantil, nivel, segundos):
    serie = datos.cargar_serie(ruta)
    tabla = features.armar_features(serie, HORIZONTE).dropna()
    cols = features.columnas_features(tabla)
    futuro = features.armar_features_futuro(serie, HORIZONTE)

    nombre = mejor_modelo(tabla)
    print(f"Para la media uso {nombre}, que gano un backtest de 4 semanas.\n")
    media = modelos.crear_modelos(cols)[nombre].fit(tabla[cols], tabla["llamadas"])
    # el escenario alto siempre es el boosting cuantil: es el unico de los cinco
    # que puede minimizar la pinball loss
    alto = modelos.modelo_cuantil(cuantil).fit(tabla[cols], tabla["llamadas"])

    salida = pd.DataFrame(index=futuro.index)
    salida["pronostico"] = np.clip(media.predict(futuro[cols]), 0, None).round(1)
    # como son dos modelos separados a veces el cuantil queda apenas por debajo de la media,
    # en ese caso me quedo con la media
    salida[f"p{int(cuantil * 100)}"] = np.maximum(
        np.clip(alto.predict(futuro[cols]), 0, None).round(1), salida["pronostico"])

    # AHT: uso la mediana de las ultimas 4 semanas
    aht = serie["aht_seg"].iloc[-4 * 168:].median()
    salida["operadores"] = [erlang.operadores_necesarios(x, aht, nivel, segundos)
                            for x in salida[f"p{int(cuantil * 100)}"]]
    # con la central cerrada no hace falta nadie
    abiertas = metricas.horas_de_atencion(tabla)
    cerrado = ~(salida.index.day_of_week * 24 + salida.index.hour).isin(abiertas)
    salida.loc[cerrado, "operadores"] = 0

    Path("resultados").mkdir(exist_ok=True)
    salida.to_csv("resultados/pronostico_semana.csv")
    print(salida[salida["pronostico"] > 0.5].to_string())
    print(f"\nAHT usado: {aht:.0f} seg. Guardado en resultados/pronostico_semana.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pronostico de llamadas del call center")
    parser.add_argument("accion", choices=["simular", "evaluar", "pronosticar"])
    parser.add_argument("--datos", help="ruta al export de llamadas (csv)")
    parser.add_argument("--semanas", type=int, default=8, help="semanas de prueba en el backtest")
    parser.add_argument("--cuantil", type=float, default=0.8)
    parser.add_argument("--nivel", type=float, default=0.8, help="nivel de servicio objetivo")
    parser.add_argument("--segundos", type=int, default=20, help="tiempo objetivo de atencion")
    args = parser.parse_args()

    if args.accion == "simular":
        print("Datos simulados guardados en", datos.guardar_simulacion())
    elif args.accion == "evaluar":
        evaluar(elegir_datos(args.datos), args.semanas)
    else:
        pronosticar(elegir_datos(args.datos), args.cuantil, args.nivel, args.segundos)
