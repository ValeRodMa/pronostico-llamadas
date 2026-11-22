import numpy as np
import pandas as pd
from pathlib import Path

from src.features import feriados_uruguay

# nombres de las columnas en el export de la central
# cuando lleguen los datos reales hay que ajustar esto segun como venga el archivo.
# ojo que los valores de estado tienen que quedar como "Atendida" / "Abandonada"
COLUMNAS = {
    "fecha_hora": "fecha_hora_entrada",
    "estado": "estado",
    "espera": "espera_seg",
    "duracion": "duracion_seg",
    "cola": "cola",
}

# columnas con datos personales, se borran antes de hacer cualquier cosa
COLUMNAS_PERSONALES = ["telefono", "id_cliente", "operador"]

# la central tambien exporta un reporte "Datos de las Colas" que ya viene
# agregado por hora (una fila por hora, sin datos personales). estos son los
# nombres de las columnas que necesito de ese reporte; el resto se ignora
COLUMNAS_AGREGADO = {
    "fecha_hora": "Hora y día",
    "llamadas": "Recibidas",
    "atendidas": "Atendidas",
    "abandonadas": "Perdidas",
    "aht": "Duracion promedio atendidas",
    "espera_atendidas": "Espera promedio atendidas",
    "espera_perdidas": "Espera promedio perdidas",
}

RUTA_REAL = "data/raw/llamadas.csv"
RUTA_SIMULADA = "data/simulado/llamadas_simuladas.csv"


def es_agregado(columnas):
    """True si el archivo es el reporte por hora de la central."""
    return set(COLUMNAS_AGREGADO.values()).issubset(set(columnas))


def cargar_export(ruta):
    # los personales como texto, si no el telefono pierde el 0 de adelante
    df = pd.read_csv(ruta, dtype={c: str for c in COLUMNAS_PERSONALES})
    df = df.rename(columns={v: k for k, v in COLUMNAS.items()})
    # si la fecha viene como dd/mm/aaaa hay que agregar dayfirst=True
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    # si el export trae salientes, aca habria que filtrar y quedarse con las entrantes
    return df.sort_values("fecha_hora").reset_index(drop=True)


def anonimizar(df):
    a_borrar = [c for c in COLUMNAS_PERSONALES if c in df.columns]
    return df.drop(columns=a_borrar)


def agregar_por_hora(df):
    df = df.copy()
    df["hora"] = df["fecha_hora"].dt.floor("h")
    df["atendida"] = (df["estado"] == "Atendida").astype(int)
    df["abandonada"] = (df["estado"] == "Abandonada").astype(int)

    # llamadas = todas las que entraron (atendidas + abandonadas)
    # si contara solo las atendidas estaria subestimando la demanda justo cuando falto gente
    por_hora = df.groupby("hora").agg(
        llamadas=("fecha_hora", "size"),
        atendidas=("atendida", "sum"),
        abandonadas=("abandonada", "sum"),
        espera_media=("espera", "mean"),
    )
    # el AHT (tiempo medio de atencion) solo tiene sentido con las atendidas
    por_hora["aht_seg"] = df[df["atendida"] == 1].groupby("hora")["duracion"].mean()
    return _rellenar_horas(por_hora)


def _rellenar_horas(por_hora):
    # relleno las horas sin llamadas (noches, domingos) con 0 para que la serie no tenga huecos
    inicio = por_hora.index.min().floor("D")
    fin = por_hora.index.max().floor("D") + pd.Timedelta(hours=23)
    por_hora = por_hora.reindex(pd.date_range(inicio, fin, freq="h"))
    cuentas = ["llamadas", "atendidas", "abandonadas"]
    por_hora[cuentas] = por_hora[cuentas].fillna(0).astype(int)
    por_hora.index.name = "fecha_hora"
    return por_hora


def cargar_agregado(ruta):
    """Carga el reporte por hora de la central y lo deja igual que agregar_por_hora."""
    df = pd.read_csv(ruta)
    df = df.rename(columns={v: k for k, v in COLUMNAS_AGREGADO.items()})
    # la fecha viene como "01/11/2025, 08:00" (dia/mes/año). el reporte ademas
    # trae filas "Total": lo que no parsea como fecha, se tira
    fechas = pd.to_datetime(df["fecha_hora"], format="%d/%m/%Y, %H:%M", errors="coerce")
    if fechas.isna().all():
        # por si algun export viene con la fecha en otro formato razonable
        fechas = pd.to_datetime(df["fecha_hora"], errors="coerce", dayfirst=True)
    df["fecha_hora"] = fechas
    df = df.dropna(subset=["fecha_hora"])

    # los tiempos vienen como texto "hh:mm:ss". los paso a segundos y los convierto en
    # totales por fila (promedio por cantidad), que si se pueden sumar entre filas
    aht = pd.to_timedelta(df["aht"]).dt.total_seconds()
    esp_at = pd.to_timedelta(df["espera_atendidas"]).dt.total_seconds()
    esp_pe = pd.to_timedelta(df["espera_perdidas"]).dt.total_seconds()
    df["aht_total"] = aht * df["atendidas"]
    df["espera_total"] = esp_at * df["atendidas"] + esp_pe * df["abandonadas"]

    # el mismo reporte puede venir con varias filas para la misma hora (una por cola,
    # si se exportaron todas juntas): las cuentas se suman y los tiempos se vuelven
    # a promediar ponderando por la cantidad de llamadas de cada fila
    por_hora = df.groupby("fecha_hora")[
        ["llamadas", "atendidas", "abandonadas", "aht_total", "espera_total"]].sum()
    # un 00:00:00 en horas sin atendidas no es un AHT real, es un hueco
    atendidas = por_hora["atendidas"].replace(0, np.nan)
    por_hora["aht_seg"] = por_hora["aht_total"] / atendidas
    total = (por_hora["atendidas"] + por_hora["abandonadas"]).replace(0, np.nan)
    por_hora["espera_media"] = por_hora["espera_total"] / total
    # mismo orden de columnas que agregar_por_hora, para que todo se vea igual
    por_hora = por_hora[["llamadas", "atendidas", "abandonadas", "espera_media", "aht_seg"]]
    return _rellenar_horas(por_hora)


def cargar_serie(ruta):
    # el archivo puede venir en dos formatos: el reporte ya agregado por hora
    # de la central, o un export con una fila por llamada (como el simulado),
    # que hay que anonimizar y agregar
    if es_agregado(pd.read_csv(ruta, nrows=0).columns):
        return cargar_agregado(ruta)
    return agregar_por_hora(anonimizar(cargar_export(ruta)))


# ------------------------------------------------------------------
# datos simulados, para poder armar todo mientras no tengo los reales
# ------------------------------------------------------------------

def perfil_diario(hora, dia_semana):
    # dos picos (media mañana y media tarde) con una baja al mediodia
    manana = np.exp(-((hora - 10.5) ** 2) / (2 * 1.6 ** 2))
    tarde = 0.85 * np.exp(-((hora - 16) ** 2) / (2 * 2.0 ** 2))
    perfil = 0.15 + manana + tarde

    # horario de atencion: L a V 8-22, sabado 9-19, domingo 10-18
    abre = np.select([dia_semana <= 4, dia_semana == 5], [8, 9], 10)
    cierra = np.select([dia_semana <= 4, dia_semana == 5], [22, 19], 18)
    return perfil * ((hora >= abre) & (hora < cierra))


def simular_llamadas(inicio="2025-11-03", fin="2026-11-15", semilla=42):
    rng = np.random.default_rng(semilla)
    horas = pd.date_range(inicio, pd.Timestamp(fin) + pd.Timedelta(hours=23), freq="h")
    hora = horas.hour.to_numpy()
    dia = horas.day_of_week.to_numpy()
    fecha = horas.normalize()

    factor_dia = np.array([1.25, 1.10, 1.0, 0.97, 0.90, 0.50, 0.30])[dia]
    factor_mes = horas.month.map({1: 0.85, 2: 0.90, 3: 1.10, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.05,
                                  8: 1.0, 9: 1.0, 10: 1.0, 11: 1.05, 12: 1.15}).to_numpy()
    tendencia = 1 + 0.15 * np.arange(len(horas)) / len(horas)
    base = 42 * perfil_diario(hora, dia) * factor_dia * factor_mes * tendencia

    # feriados
    no_lab, lab, turismo = feriados_uruguay(range(horas.year.min(), horas.year.max() + 1))
    factor_feriado = np.ones(len(horas))
    factor_feriado[fecha.isin(list(no_lab))] = 0.2
    factor_feriado[fecha.isin(list(turismo))] = 0.45
    factor_feriado[fecha.isin(list(lab))] = 0.85
    despues = (fecha - pd.Timedelta(days=1)).isin(list(no_lab | turismo)) & ~fecha.isin(list(no_lab | turismo))
    factor_feriado[despues] = 1.2

    # ruido por dia (hay dias que simplemente llaman mas) y algunos incidentes
    dias_unicos = fecha.unique()
    ruido_dia = pd.Series(rng.lognormal(0, 0.07, len(dias_unicos)), index=dias_unicos)
    habiles = [d for d in dias_unicos if d.day_of_week < 5]
    incidentes = rng.choice(habiles, size=8, replace=False)
    ruido_dia.loc[incidentes] *= rng.uniform(1.6, 2.3, size=8)
    factor_ruido = ruido_dia.loc[fecha].to_numpy()

    lam = base * factor_feriado * factor_ruido
    cantidad = rng.poisson(lam)

    # la central se dimensiona con lo "normal", entonces cuando hay mas llamadas de lo esperado
    # aumenta la espera y la gente corta
    plan = np.maximum(base * factor_feriado, 1)
    sobrecarga = np.repeat(cantidad / plan, cantidad)

    n = cantidad.sum()
    t = np.repeat(horas.to_numpy(), cantidad) + pd.to_timedelta(rng.integers(0, 3600, n), unit="s").to_numpy()
    cola = rng.choice(["Consultas", "Reclamos", "Ventas"], size=n, p=[0.6, 0.25, 0.15])

    p_abandono = np.clip(0.03 + 0.3 * np.maximum(sobrecarga - 0.95, 0), 0, 0.5)
    abandona = rng.random(n) < p_abandono
    estado = np.where(abandona, "Abandonada", "Atendida")

    espera_media = 15 + 120 * np.maximum(sobrecarga - 0.9, 0)
    espera = np.where(abandona, rng.exponential(45, n), rng.exponential(espera_media))

    media_dur = pd.Series(cola).map({"Consultas": 180, "Reclamos": 300, "Ventas": 240}).to_numpy()
    sigma = 0.5
    duracion = rng.lognormal(np.log(media_dur) - sigma ** 2 / 2, sigma)
    duracion = np.where(abandona, 0, duracion)

    df = pd.DataFrame({
        "fecha_hora_entrada": pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S"),
        "telefono": ["09" + str(x) for x in rng.integers(1_000_000, 9_999_999, n)],
        "id_cliente": rng.integers(100_000, 999_999, n),
        "cola": cola,
        "estado": estado,
        "espera_seg": espera.round().astype(int),
        "duracion_seg": duracion.round().astype(int),
        "operador": np.where(abandona, "", ["OP" + str(x).zfill(2) for x in rng.integers(1, 26, n)]),
    })
    return df.sort_values("fecha_hora_entrada").reset_index(drop=True)


def guardar_simulacion(ruta=RUTA_SIMULADA):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    simular_llamadas().to_csv(ruta, index=False)
    return ruta
