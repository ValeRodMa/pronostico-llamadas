import pandas as pd

from src import datos

# un pedacito del reporte real de la central, con un hueco a las 10 (esa hora
# no vino en el archivo) para probar el relleno, y con la fila "Total" que el
# reporte trae al final y que hay que tirar
REPORTE = pd.DataFrame({
    "Hora y día": ["01/11/2025, 08:00", "01/11/2025, 09:00", "01/11/2025, 11:00", "Total"],
    "Recibidas": [5, 10, 31, 46],
    "Atendidas": [5, 10, 29, 44],
    "Perdidas": [0, 0, 2, 2],
    "Duracion promedio atendidas": ["00:02:07", "00:02:34", "00:02:30", "00:02:24"],
    "Espera promedio atendidas": ["00:00:08", "00:00:14", "00:00:20", "00:00:15"],
    "Espera promedio perdidas": ["00:00:00", "00:00:00", "00:00:02", "00:00:02"],
})


def test_detecta_el_formato():
    # el reporte por hora se reconoce por sus columnas, el export por llamada no
    assert datos.es_agregado(REPORTE.columns)
    assert not datos.es_agregado(datos.COLUMNAS.values())


def test_cargar_agregado(tmp_path):
    ruta = tmp_path / "reporte.csv"
    REPORTE.to_csv(ruta, index=False)
    serie = datos.cargar_agregado(ruta)

    # la fecha es dia/mes: 01/11 tiene que ser noviembre, no enero
    assert serie.index.min() == pd.Timestamp("2025-11-01 00:00")
    # el dia se rellena entero, incluida la hora 10 que faltaba en el archivo
    assert len(serie) == 24
    assert serie.loc["2025-11-01 10:00", "llamadas"] == 0

    fila = serie.loc["2025-11-01 08:00"]
    assert fila["llamadas"] == 5
    assert fila["aht_seg"] == 127  # "00:02:07" en segundos
    assert fila["espera_media"] == 8

    # a las 11 la espera es el promedio ponderado de atendidas y perdidas
    espera_11 = serie.loc["2025-11-01 11:00", "espera_media"]
    assert abs(espera_11 - (20 * 29 + 2 * 2) / 31) < 1e-9

    # mismas columnas y en el mismo orden que la agregacion por llamada,
    # para que el resto del pipeline no note la diferencia
    assert list(serie.columns) == ["llamadas", "atendidas", "abandonadas", "espera_media", "aht_seg"]


def test_cargar_agregado_con_varias_colas(tmp_path):
    # si se exportan todas las colas juntas, la misma hora aparece varias veces
    # (una fila por cola): hay que sumar las cantidades y reponderar los tiempos
    dos_colas = pd.DataFrame({
        "Hora y día": ["01/11/2025, 08:00", "01/11/2025, 08:00", "Total"],
        "Recibidas": [10, 5, 15],
        "Atendidas": [10, 4, 14],
        "Perdidas": [0, 1, 1],
        "Duracion promedio atendidas": ["00:01:00", "00:02:30", "00:01:26"],
        "Espera promedio atendidas": ["00:00:10", "00:00:20", "00:00:13"],
        "Espera promedio perdidas": ["00:00:00", "00:00:30", "00:00:30"],
    })
    ruta = tmp_path / "reporte.csv"
    dos_colas.to_csv(ruta, index=False)
    serie = datos.cargar_agregado(ruta)

    fila = serie.loc["2025-11-01 08:00"]
    assert fila["llamadas"] == 15
    assert fila["atendidas"] == 14
    assert fila["abandonadas"] == 1
    # aht ponderado: (60*10 + 150*4) / 14
    assert abs(fila["aht_seg"] - (60 * 10 + 150 * 4) / 14) < 1e-9
    # espera ponderada sobre las 15 llamadas: (10*10 + 20*4 + 30*1) / 15
    assert abs(fila["espera_media"] - (10 * 10 + 20 * 4 + 30 * 1) / 15) < 1e-9
