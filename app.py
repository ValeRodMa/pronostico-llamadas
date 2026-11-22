import inspect
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance

from src import datos, erlang, exploracion, features, metricas, modelos, validacion

st.set_page_config(page_title="Pronóstico de llamadas", layout="wide")

# cuando streamlit recalcula, por defecto deja el contenido viejo medio gris de fondo
# mientras arma el nuevo, y eso confunde: parece que la pagina ya cargo cuando todavia
# esta calculando. con este css lo desactualizado se oculta y solo queda el cartel de
# "cargando" de cada calculo (los show_spinner de abajo)
st.markdown(
    "<style>div[data-stale='true'], .stale-element { visibility: hidden !important; }</style>",
    unsafe_allow_html=True,
)

HORIZONTE = 168
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
COLORES = {
    "real": "#2b2b2b",
    "Naive estacional": "#9e9e9e",
    "Holt-Winters": "#9467bd",
    "Ridge": "#1f77b4",
    "Random Forest": "#2ca02c",
    "Gradient Boosting": "#d62728",
}
DESCRIPCIONES = {
    "hora": "Hora del día",
    "dia_semana": "Día de la semana (0 = lunes)",
    "hora_semana": "Hora de la semana (0 a 167)",
    "mes": "Mes",
    "feriado": "Es feriado no laborable, carnaval o turismo",
    "feriado_laborable": "Es feriado laborable",
    "post_feriado": "Es el día después de un feriado",
    "t": "Días desde el inicio (tendencia)",
    "media_misma_hora_4sem": "Promedio de la misma hora en las últimas 4 semanas",
    "nivel_semana": "Promedio de la última semana conocida",
}


# ---------- carga y calculos (con cache para que no se recalcule todo en cada click) ----------

def formato_agregado(ruta):
    # el reporte real de la central ya viene agregado por hora; el simulado
    # siempre es por llamada, asi que para ese ni miro el encabezado
    return ruta != datos.RUTA_SIMULADA and datos.es_agregado(pd.read_csv(ruta, nrows=0).columns)


@st.cache_data(show_spinner="Cargando: leyendo el export...")
def cargar_crudo(ruta):
    # si todavia no existen los datos simulados (primera vez que se abre la app), los genero
    if ruta == datos.RUTA_SIMULADA and not Path(ruta).exists():
        datos.guardar_simulacion()
    return datos.cargar_export(ruta)


@st.cache_data(show_spinner="Cargando: armando la serie por hora...")
def cargar_serie(ruta):
    if formato_agregado(ruta):
        return datos.cargar_agregado(ruta)
    return datos.agregar_por_hora(datos.anonimizar(cargar_crudo(ruta)))


@st.cache_data(show_spinner="Cargando: armando la tabla de features...")
def cargar_tabla(ruta, horizonte=HORIZONTE):
    return features.armar_features(cargar_serie(ruta), horizonte).dropna()


@st.cache_data(show_spinner="Cargando: entrenando los 5 modelos semana por semana (la primera vez tarda alrededor de un minuto)...")
def correr_backtest(ruta, semanas):
    tabla = cargar_tabla(ruta)
    mods = modelos.crear_modelos(features.columnas_features(tabla))
    return validacion.backtest(tabla, mods, semanas, HORIZONTE)


@st.cache_data(show_spinner="Cargando: comparando las dos formas de validar...")
def correr_comparacion(ruta):
    return validacion.comparar_validaciones(cargar_tabla(ruta), modelos.gradient_boosting())


@st.cache_data(show_spinner="Cargando: entrenando un modelo por cuantil...")
def correr_cuantiles(ruta, cuantiles):
    # un modelo por cuantil, entrenado sin las ultimas 4 semanas: la cobertura hay que
    # medirla en datos que el modelo no vio, si no da bien de puro memorioso
    tabla = cargar_tabla(ruta)
    cols = features.columnas_features(tabla)
    corte = len(tabla) - 4 * HORIZONTE
    ent, prueba = tabla.iloc[:corte], tabla.iloc[corte:]
    pred = pd.DataFrame({"real": prueba["llamadas"], "hora_semana": prueba["hora_semana"]}, index=prueba.index)
    for q in cuantiles:
        m = modelos.modelo_cuantil(q).fit(ent[cols], ent["llamadas"])
        pred[q] = np.clip(m.predict(prueba[cols]), 0, None)
    return pred


@st.cache_data(show_spinner="Cargando: calculando la importancia de cada variable...")
def correr_importancia(ruta):
    # importancia por permutacion: desordeno una columna por vez y mido cuanto empeora el MAE.
    # si mezclar una variable no cambia nada, es que el modelo no la estaba usando
    tabla = cargar_tabla(ruta)
    cols = features.columnas_features(tabla)
    corte = len(tabla) - 4 * HORIZONTE
    ent, prueba = tabla.iloc[:corte], tabla.iloc[corte:]
    m = modelos.gradient_boosting().fit(ent[cols], ent["llamadas"])
    r = permutation_importance(m, prueba[cols], prueba["llamadas"], n_repeats=5, random_state=0,
                               scoring="neg_mean_absolute_error")
    return pd.Series(r.importances_mean, index=cols).sort_values()


@st.cache_data(show_spinner="Cargando: estimando la paciencia del histórico...")
def paciencia_media(ruta):
    # cuanto aguanta la gente antes de cortar: la mejor pista que tengo es la espera
    # promedio de las llamadas que se perdieron. es una medida censurada (solo veo
    # cuanto espero el que corto, no cuanto mas habria aguantado), pero como default sirve
    if formato_agregado(ruta):
        df = pd.read_csv(ruta)
        fechas = pd.to_datetime(df[datos.COLUMNAS_AGREGADO["fecha_hora"]],
                                format="%d/%m/%Y, %H:%M", errors="coerce")
        df = df[fechas.notna()]  # fuera la fila Total, igual que en datos.py
        esp = pd.to_timedelta(df[datos.COLUMNAS_AGREGADO["espera_perdidas"]],
                              errors="coerce").dt.total_seconds()
        n = pd.to_numeric(df[datos.COLUMNAS_AGREGADO["abandonadas"]], errors="coerce")
        bien = (n > 0) & esp.notna()
        if bien.any():
            # promedio ponderado por la cantidad de perdidas de cada hora
            return float((esp[bien] * n[bien]).sum() / n[bien].sum())
    else:
        perdidas = cargar_crudo(ruta).query("estado == 'Abandonada'")
        if len(perdidas):
            return float(perdidas["espera"].mean())
    return 60.0


@st.cache_data(show_spinner="Cargando: entrenando con todos los datos para el pronóstico final...")
def correr_pronostico(ruta, cuantil, nombre_modelo, horizonte=HORIZONTE):
    serie = cargar_serie(ruta)
    tabla = cargar_tabla(ruta, horizonte)
    cols = features.columnas_features(tabla)
    futuro = features.armar_features_futuro(serie, horizonte)
    # para la media uso el modelo que gano el backtest del paso 6: pronosticar con uno fijo
    # mientras la tabla muestra que gano otro no se podria defender
    modelo = modelos.crear_modelos(cols)[nombre_modelo]
    media = modelo.fit(tabla[cols], tabla["llamadas"]).predict(futuro[cols])
    # el escenario alto siempre es el boosting cuantil, el unico que minimiza la pinball loss
    alto = modelos.modelo_cuantil(cuantil).fit(tabla[cols], tabla["llamadas"]).predict(futuro[cols])
    media = np.clip(media, 0, None)
    alto = np.maximum(np.clip(alto, 0, None), media)
    return pd.DataFrame({"pronostico": media, "alto": alto}, index=futuro.index)


def mejor_modelo(ruta, semanas=8):
    # el ganador del mismo backtest que muestra la pagina 6 (esta cacheado, no entrena de nuevo)
    res = correr_backtest(ruta, semanas)
    abiertas = solo_horas_abiertas(res, ruta)
    nombres = [c for c in res.columns if c not in ("real", "semana")]
    return metricas.tabla_metricas(abiertas, nombres).iloc[0]["Modelo"]


@st.cache_data(show_spinner="Cargando: midiendo el error a 1, 2, 3 y 4 semanas (tarda unos minutos la primera vez)...")
def error_por_horizonte(ruta, nombre_modelo, semanas_prueba=4):
    # backtest corto del ganador y del naive con 1 a 4 semanas de anticipacion.
    # las semanas de prueba son las mismas en los cuatro casos (ventanas de 168 h al
    # final de la serie); lo que cambia es la tabla: a mas horizonte, menos lags.
    # son 4 rondas y no mas porque esto entrena 4 tablas distintas y se hace largo
    filas = []
    for sem in (1, 2, 3, 4):
        tabla = cargar_tabla(ruta, sem * HORIZONTE)
        mods = modelos.crear_modelos(features.columnas_features(tabla))
        elegidos = {"Naive estacional": mods["Naive estacional"]}
        elegidos.setdefault(nombre_modelo, mods[nombre_modelo])
        res = validacion.backtest(tabla, elegidos, semanas_prueba, HORIZONTE)
        ab = solo_horas_abiertas(res.join(tabla[["hora_semana"]]), ruta)
        fila = {"Semanas de anticipación": sem}
        for nombre in elegidos:
            fila[nombre] = metricas.mae(ab["real"], ab[nombre])
        filas.append(fila)
    return pd.DataFrame(filas).set_index("Semanas de anticipación")


def solo_horas_abiertas(df, ruta):
    # me quedo solo con las horas en que la central suele estar abierta: el error en las
    # horas cerradas es trivial (todos predicen 0) y abarataria las metricas
    abiertas = metricas.horas_de_atencion(cargar_tabla(ruta))
    hs = df["hora_semana"] if "hora_semana" in df else cargar_tabla(ruta).loc[df.index, "hora_semana"]
    return df[hs.isin(abiertas)]


def marcar_momento(fig, x, texto, color="#d62728"):
    # add_vline con fechas a veces falla con la anotacion, asi que la agrego aparte
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=color, dash="dot"))
    fig.add_annotation(x=x, y=1, yref="paper", text=texto, showarrow=False, yanchor="bottom",
                       font=dict(color=color))


def ver_codigo(*funciones):
    # cada pagina muestra el codigo que de verdad ejecuta ese paso: lo saco del
    # modulo con inspect, no es una copia pegada, asi nunca queda desactualizado
    with st.expander("Ver el código de este paso"):
        for f in funciones:
            st.caption("src/" + f.__module__.split(".")[-1] + ".py")
            st.code(inspect.getsource(f), language="python")


# ---------- paginas ----------

def inicio(ruta, simulados):
    st.title("Pronóstico de llamadas de un call center")
    st.write(
        "El objetivo es predecir cuántas llamadas van a entrar en cada hora de la semana que viene, "
        "para poder planificar cuántos operadores poner en cada franja. "
        "La app recorre todo el proceso, desde el export de la central hasta el pronóstico."
    )
    if simulados:
        st.info("Estás viendo datos simulados. Para usar los reales, subí el export de la central "
                "desde el menú de la izquierda (o poné el archivo en data/raw/llamadas.csv).")

    serie = cargar_serie(ruta)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Llamadas", f"{serie['llamadas'].sum():,}".replace(",", "."))
    c2.metric("Horas de datos", f"{len(serie):,}".replace(",", "."))
    c3.metric("Desde", serie.index.min().strftime("%d/%m/%Y"))
    c4.metric("Hasta", serie.index.max().strftime("%d/%m/%Y"))

    st.subheader("Recorrido")
    st.markdown(
        "1. **Datos crudos**: el export tal cual llega de la central.\n"
        "2. **Anonimización**: saco los datos personales (si trae) y dejo la serie por hora.\n"
        "3. **Exploración**: tendencia, estacionalidad diaria y semanal, autocorrelación.\n"
        "4. **De serie a tabla**: cómo se arma la tabla que usan los modelos, sin usar datos del futuro.\n"
        "5. **Validación**: por qué no sirve el k-fold común y cómo funciona el walk-forward.\n"
        "6. **Modelos**: baseline, Holt-Winters, Ridge, Random Forest y Gradient Boosting.\n"
        "7. **Incertidumbre**: regresión cuantil para tener un escenario alto.\n"
        "8. **Pronóstico**: la semana que viene y cuántos operadores hacen falta."
    )
    st.info(
        "Cómo leer la app: cada página arranca con la idea en criollo, después muestra los "
        "resultados, y al final tiene un desplegable con el código real que ejecuta ese paso. "
        "Cuando algo dice «Cargando...» es porque está entrenando modelos de verdad: la primera "
        "vez tarda, después queda guardado y va al toque."
    )


def pagina_crudos(ruta, simulados):
    st.header("1. Los datos crudos")
    st.info(
        "**La idea:** mirar lo que llega antes de tocar nada. De este archivo me importan dos "
        "números por hora: cuántas llamadas entraron (lo que voy a predecir) y cuánto dura una "
        "atención (lo que necesita la fórmula de Erlang al final para pasar de llamadas a gente)."
    )
    if formato_agregado(ruta):
        # el reporte real no es por llamada: ya viene una fila por hora
        st.write("Así llega el reporte de la central: una fila por hora, ya agregado "
                 "(recibidas, atendidas, perdidas y tiempos promedio).")
        st.markdown(
            "- **Hora y día**: la franja horaria del registro.\n"
            "- **Recibidas**: todas las llamadas que entraron en esa hora (atendidas + perdidas). "
            "Esto es lo que el modelo va a aprender a predecir.\n"
            "- **Atendidas / Perdidas**: cómo terminó cada llamada. Las perdidas son personas que se "
            "cansaron de esperar y cortaron.\n"
            "- **Duración promedio atendidas**: el AHT, cuánto dura una atención. Lo necesita la "
            "fórmula de Erlang en el paso 8 para pasar de llamadas a operadores.\n"
            "- **Espera promedio**: cuánto esperó la gente antes de ser atendida (o de cortar)."
        )
        crudo = pd.read_csv(ruta)
        serie = cargar_serie(ruta)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Llamadas", f"{serie['llamadas'].sum():,}".replace(",", "."))
        c2.metric("Abandonadas", f"{100 * serie['abandonadas'].sum() / serie['llamadas'].sum():.1f} %")
        c3.metric("Horas con registro", f"{len(crudo):,}".replace(",", "."))
        c4.metric("Duración media (atendidas)", f"{serie['aht_seg'].mean():.0f} seg")
        st.dataframe(crudo.head(300), hide_index=True)
        st.caption("Este formato ya viene sin datos personales. Los tiempos (hh:mm:ss) "
                   "se convierten a segundos en el paso 2.")
        totales = pd.DataFrame({"estado": ["Atendidas", "Perdidas"],
                                "llamadas": [serie["atendidas"].sum(), serie["abandonadas"].sum()]})
        c1, _ = st.columns(2)
        c1.plotly_chart(px.bar(totales, x="estado", y="llamadas", title="Llamadas por estado",
                               labels={"llamadas": "Llamadas", "estado": ""}))
        st.write(
            "De este archivo salen las dos cosas que alimentan todo lo demás: cuántas llamadas entran "
            "por hora (el target del modelo) y cuánto dura una atención (el insumo para calcular "
            "operadores). La app reconoce este formato por los nombres de las columnas; el desplegable "
            "muestra la función que lo decide."
        )
        ver_codigo(datos.es_agregado)
        return

    st.write("Así llega el export de la central: una fila por cada llamada entrante.")
    st.markdown(
        "- **fecha_hora_entrada**: cuándo entró la llamada. Es lo único imprescindible para armar la "
        "serie por hora.\n"
        "- **estado**: si fue atendida o la persona cortó antes (abandonada).\n"
        "- **espera_seg / duracion_seg**: segundos de espera en cola y de conversación.\n"
        "- **cola**: a qué cola entró (consultas, reclamos, ventas).\n"
        "- **telefono, id_cliente, operador**: datos personales. No aportan nada para predecir volumen "
        "y se borran en el paso 2."
    )
    crudo = cargar_crudo(ruta)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Llamadas", f"{len(crudo):,}".replace(",", "."))
    c2.metric("Abandonadas", f"{100 * (crudo['estado'] == 'Abandonada').mean():.1f} %")
    c3.metric("Espera media", f"{crudo['espera'].mean():.0f} seg")
    c4.metric("Duración media (atendidas)", f"{crudo.loc[crudo['estado'] == 'Atendida', 'duracion'].mean():.0f} seg")

    st.dataframe(crudo.head(300), hide_index=True)
    aviso = "Las columnas telefono, id_cliente y operador son datos personales y se borran en el paso 2."
    if simulados:
        aviso += " En los datos simulados son inventados."
    st.caption(aviso)

    c1, c2 = st.columns(2)
    por_estado = crudo["estado"].value_counts().reset_index()
    c1.plotly_chart(px.bar(por_estado, x="estado", y="count", title="Llamadas por estado",
                           labels={"count": "Llamadas", "estado": ""}))
    if "cola" in crudo:
        por_cola = crudo["cola"].value_counts().reset_index()
        c2.plotly_chart(px.bar(por_cola, x="cola", y="count", title="Llamadas por cola",
                               labels={"count": "Llamadas", "cola": ""}))
    ver_codigo(datos.cargar_export)


def pagina_anonimizacion(ruta, simulados):
    st.header("2. Anonimización y agregación por hora")
    st.info(
        "**La idea:** dejar los datos limpios, sin nada personal y en el formato que necesito: "
        "una fila por hora. Y una decisión que parece menor pero no lo es: cuento **todas** las "
        "llamadas que entraron, también las que se cortaron — si contara solo las atendidas, el "
        "modelo aprendería que «hay poca demanda» justo en las horas donde faltó gente."
    )
    if formato_agregado(ruta):
        st.write(
            "Este reporte ya viene agregado por hora y **sin datos personales**, así que acá no hay "
            "nada que borrar (el paso queda implementado por si algún día llega un export por llamada). "
            "Lo que sí se hace: renombrar columnas, convertir los tiempos hh:mm:ss a segundos y "
            "rellenar con 0 las horas sin registro."
        )
        c1, c2 = st.columns(2)
        c1.markdown("**Columnas del reporte**")
        c1.code("\n".join(pd.read_csv(ruta, nrows=0).columns))
        c2.markdown("**Columnas de la serie**")
        c2.code("\n".join(cargar_serie(ruta).columns))
        st.subheader("La serie por hora")
        st.write("Las horas sin registro (por ejemplo la madrugada) quedan en 0, para que la serie no tenga huecos.")
        ver_codigo(datos.cargar_agregado, datos._rellenar_horas)
    else:
        crudo = cargar_crudo(ruta)
        anon = datos.anonimizar(crudo)

        c1, c2 = st.columns(2)
        c1.markdown("**Columnas del export**")
        c1.code("\n".join(crudo.columns))
        c2.markdown("**Después de anonimizar**")
        c2.code("\n".join(anon.columns))
        st.write(
            "Lo primero es borrar todo lo que identifica a una persona. Para predecir cuántas llamadas "
            "van a entrar no hace falta saber quién llamó."
        )

        st.subheader("De llamadas a una serie por hora")
        st.write(
            "Después cuento cuántas llamadas entraron en cada hora. Las horas sin llamadas (por ejemplo "
            "la madrugada) quedan en 0, para que la serie no tenga huecos."
        )
        ver_codigo(datos.anonimizar, datos.agregar_por_hora, datos._rellenar_horas)
    serie = cargar_serie(ruta)
    st.dataframe(serie.iloc[8:56].round(1))
    st.caption(
        "Cada fila es una hora. llamadas = atendidas + abandonadas (la demanda completa); espera_media "
        "y aht_seg están en segundos; aht_seg queda vacío en las horas sin atendidas, porque ahí no hay "
        "ninguna duración para promediar."
    )

    st.subheader("Atendidas + abandonadas = demanda real")
    st.write(
        "Para el modelo cuento **todas** las llamadas que entraron, no solo las atendidas. "
        "Si contara solo las atendidas, en los días con pocos operadores la demanda parecería más baja "
        "de lo que fue, y el modelo aprendería a pedir menos gente justo cuando más falta."
    )
    por_dia = serie["abandonadas"].resample("D").sum()
    default = por_dia.idxmax().date()
    dia = st.date_input("Día", value=default, min_value=serie.index.min().date(),
                        max_value=serie.index.max().date())
    un_dia = serie.loc[str(dia)]
    fig = go.Figure()
    fig.add_bar(x=un_dia.index.hour, y=un_dia["atendidas"], name="Atendidas", marker_color="#1f77b4")
    fig.add_bar(x=un_dia.index.hour, y=un_dia["abandonadas"], name="Abandonadas", marker_color="#ff7f0e")
    fig.update_layout(barmode="stack", xaxis_title="Hora", yaxis_title="Llamadas",
                      title=f"Llamadas del {dia.strftime('%d/%m/%Y')}")
    st.plotly_chart(fig)
    st.caption("Cada barra es una hora del día elegido: lo naranja es demanda que existió igual, "
               "aunque nadie la haya atendido. Si el modelo no la contara, aprendería que esos días "
               "'hubo pocas llamadas' cuando en realidad faltó gente para atenderlas.")
    if dia == default:
        st.caption("Por defecto muestra el día con más abandonos del período.")


def pagina_exploracion(ruta, simulados):
    st.header("3. Exploración")
    st.info(
        "**La idea:** mirar la serie antes de modelar. Lo que se **repite** (el patrón del día "
        "y de la semana) es lo que un modelo puede aprender; lo que no se repite (picos "
        "aislados, incidentes) no lo va a acertar nadie. De esta página salen las features del "
        "paso 4 y el baseline contra el que compito después."
    )
    st.write(
        "Antes de modelar hay que mirar la serie: de esta página salen las features del paso 4 y el "
        "baseline contra el que se compara todo lo demás. Busco tres cosas: si la serie crece o baja "
        "(tendencia), qué patrones se repiten (estacionalidad) y cuánto se parece el presente al "
        "pasado (autocorrelación)."
    )
    serie = cargar_serie(ruta)

    vista = st.radio("Ver la serie", ["Por día", "Por hora"], horizontal=True)
    if vista == "Por día":
        diaria = serie["llamadas"].resample("D").sum()
        fig = px.line(x=diaria.index, y=diaria.values, labels={"x": "", "y": "Llamadas por día"})
        fer = features.variables_calendario(diaria.index)
        fer = diaria[fer["feriado"] == 1]
        fig.add_scatter(x=fer.index, y=fer.values, mode="markers", name="Feriados / turismo",
                        marker=dict(color="#d62728", size=7))
    else:
        fig = px.line(x=serie.index, y=serie["llamadas"], labels={"x": "", "y": "Llamadas por hora"})
        fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)
    # la direccion del nivel depende de los datos (el simulado crece, el real cayo), asi que
    # la describo con los numeros de la propia serie en vez de dejar una frase fija
    vol = serie["llamadas"].resample("W").sum()
    ini, fin = vol.iloc[1:5].mean(), vol.iloc[-5:-1].mean()
    cambio = 100 * (fin / ini - 1)
    if cambio <= -10:
        nivel_txt = (f"el nivel **cayó** durante el año: las primeras semanas promedian {ini:.0f} "
                     f"llamadas y las últimas {fin:.0f} ({cambio:.0f} %)")
    elif cambio >= 10:
        nivel_txt = (f"el nivel **subió** durante el año: las primeras semanas promedian {ini:.0f} "
                     f"llamadas y las últimas {fin:.0f} (+{cambio:.0f} %)")
    else:
        nivel_txt = (f"el nivel se mantuvo bastante estable durante el año "
                     f"(primeras semanas {ini:.0f} llamadas, últimas {fin:.0f})")
    st.write(
        f"Dos cosas saltan a la vista: {nivel_txt}, y una semana que se repite, con caídas en los "
        f"feriados y algunos picos aislados que no siguen ningún patrón."
    )

    c1, c2 = st.columns(2)
    mapa = exploracion.mapa_calor(serie)
    fig = px.imshow(mapa, y=DIAS, labels=dict(x="Hora", y="", color="Llamadas"),
                    color_continuous_scale="Blues", aspect="auto", title="Promedio por día y hora")
    c1.plotly_chart(fig)

    perfil = mapa.T.copy()
    perfil.columns = DIAS
    fig = px.line(perfil, labels={"value": "Llamadas promedio", "hora": "Hora", "variable": "Día"},
                  title="Perfil del día según el día de la semana")
    c2.plotly_chart(fig)
    st.write(
        "El mapa de calor es la «semana típica»: cada celda es el promedio de llamadas de ese día a esa "
        "hora, y el gráfico de al lado muestra las mismas filas como curvas. Ahí se ve la **doble "
        "estacionalidad**: un patrón dentro del día (horas pico y horas muertas) y otro dentro de la "
        "semana (días sistemáticamente más cargados que otros). Un modelo que no capture las dos cosas "
        "a la vez no tiene chance."
    )

    st.subheader("Autocorrelación")
    acf = exploracion.autocorrelacion(serie["llamadas"], 400)
    fig = go.Figure(go.Bar(x=np.arange(1, 401), y=acf[1:], marker_color="#1f77b4"))
    for lag, texto in [(24, "24 h"), (168, "1 semana"), (336, "2 semanas")]:
        fig.add_vline(x=lag, line_dash="dot", line_color="#d62728", annotation_text=texto)
    fig.update_layout(xaxis_title="Lag (horas)", yaxis_title="Correlación", height=350)
    st.plotly_chart(fig)
    st.write(
        f"La autocorrelación mide cuánto se parece la serie a sí misma corrida k horas. "
        f"Los picos en 24 y 168 confirman lo anterior: una hora se parece a la misma hora de ayer "
        f"(correlación {acf[24]:.2f}) y todavía más a la misma hora de la semana pasada ({acf[168]:.2f}). "
        f"Por eso las features principales van a ser esos lags."
    )

    st.subheader("Qué decisiones salen de esta página")
    st.markdown(
        "- La semana pasada se parece mucho a esta → el baseline a vencer es el **naive estacional** "
        "(repetir la semana anterior tal cual), y los lags de 168, 336, 504 y 672 horas van a ser las "
        "features más importantes.\n"
        "- Hay patrón diario y semanal → features de calendario: hora, día de la semana, hora de la semana.\n"
        "- Los feriados hunden la serie → variables que marcan los feriados de Uruguay (y el día "
        "siguiente, que suele venir recargado).\n"
        "- Si la serie tiene tendencia, repetir el pasado no alcanza → una feature `t` que cuenta los "
        "días desde el inicio, para que el modelo pueda ajustar el nivel."
    )
    ver_codigo(exploracion.mapa_calor, exploracion.autocorrelacion)


def pagina_serie_a_tabla(ruta, simulados):
    st.header("4. De serie a tabla")
    st.info(
        "**La idea:** los modelos del curso comen tablas, no series. Acá cada hora pasa a ser "
        "una fila, con una regla de oro: cada fila solo puede usar información que **ya existía** "
        "en el momento de predecir. Usar algo del futuro se llama *data leakage* y es la trampa "
        "más fácil de cometer sin darse cuenta."
    )
    st.write(
        "Los modelos que vimos en el curso (Ridge, árboles, boosting) trabajan con una tabla: filas con "
        "features y un target. Para usarlos con una serie, cada hora pasa a ser una fila, y las features "
        "son cosas que **ya se sabían** al momento de hacer la predicción."
    )

    opciones = {"Una semana antes (168 h)": 168, "Un día antes (24 h)": 24}
    h = opciones[st.radio("¿Con cuánta anticipación predigo?", list(opciones), horizontal=True)]
    serie = cargar_serie(ruta)
    tabla = cargar_tabla(ruta, h)

    # por defecto un miercoles de las ultimas semanas a las 11
    d = tabla.index.max().normalize() - pd.Timedelta(days=7)
    while d.day_of_week != 2:
        d -= pd.Timedelta(days=1)
    c1, c2 = st.columns([1, 2])
    dia = c1.date_input("Día a predecir", value=d.date(), min_value=tabla.index.min().date(),
                        max_value=tabla.index.max().date())
    hora = c2.slider("Hora", 0, 23, 11)
    t = pd.Timestamp(dia) + pd.Timedelta(hours=hora)

    ventana = serie.loc[t - pd.Timedelta(hours=720): t + pd.Timedelta(hours=24), "llamadas"]
    fig = go.Figure()
    fig.add_scatter(x=ventana.index, y=ventana.values, mode="lines", line=dict(color="#b0b0b0"),
                    name="Llamadas")
    fig.add_vrect(x0=t - pd.Timedelta(hours=h), x1=t, fillcolor="#ffcc80", opacity=0.3, line_width=0)
    fig.add_annotation(x=t - pd.Timedelta(hours=h / 2), y=1, yref="paper", showarrow=False,
                       text="todavía no se conoce", yanchor="bottom")

    lags = [c for c in tabla.columns if c.startswith("lag_")]
    tiempos = [t - pd.Timedelta(hours=int(c.split("_")[1])) for c in lags]
    fig.add_scatter(x=tiempos, y=serie.loc[tiempos, "llamadas"], mode="markers+text",
                    text=[c.replace("_", " ") for c in lags], textposition="top center",
                    marker=dict(color="#1f77b4", size=10), name="Lags que usa")

    if t in tabla.index:
        fin_nivel = t - pd.Timedelta(hours=h)
        nivel = tabla.loc[t, "nivel_semana"]
        fig.add_scatter(x=[fin_nivel - pd.Timedelta(hours=168), fin_nivel], y=[nivel, nivel], mode="lines",
                        line=dict(color="#2ca02c", width=3), name="Nivel de la última semana conocida")

    fig.add_scatter(x=[t], y=[serie.loc[t, "llamadas"]], mode="markers",
                    marker=dict(symbol="star", size=16, color="#d62728"), name="Lo que quiero predecir")
    fig.update_layout(height=420, yaxis_title="Llamadas", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)
    st.write(
        f"Si predigo con {h} horas de anticipación, todo lo que pasó en la franja naranja todavía no existe "
        f"al momento de predecir. Por eso solo uso lags de {h} horas o más. Si usara, por ejemplo, el lag de 1 hora, "
        f"el backtest daría un error mucho más bajo que el que voy a tener de verdad, porque en la realidad ese "
        f"dato todavía no existe: eso es **data leakage**. Probá cambiar la anticipación acá arriba: con un "
        f"día se habilitan lags más recientes (los puntos azules se acercan a la estrella), pero una semana "
        f"es lo que el call center necesita para armar los turnos."
    )

    if t not in tabla.index:
        st.warning("Para esa hora no hay suficiente historia para armar todas las features.")
        return

    st.subheader(f"La fila de la tabla para el {t.strftime('%d/%m/%Y %H:%M')}")
    fila = tabla.loc[t]
    desc = {**DESCRIPCIONES, **{c: f"Llamadas hace {c.split('_')[1]} horas" for c in lags}}
    mostrar = pd.DataFrame({
        "Feature": fila.index,
        "Qué es": [desc.get(c, "Target: llamadas que entraron (lo que hay que predecir)") for c in fila.index],
        "Valor": fila.values.round(1),
    })
    st.dataframe(mostrar, hide_index=True, height=35 * (len(mostrar) + 1) + 3)
    st.caption(f"La tabla completa tiene {len(tabla):,} filas y {len(tabla.columns) - 1} features.".replace(",", "."))
    st.write(
        "Las features vienen de tres familias. Las de **calendario** (hora, día, mes, feriados) se "
        "conocen incluso para el futuro, así que siempre están disponibles. Los **lags** son la serie "
        "misma mirada hacia atrás: cuántas llamadas entraron hace 1, 2, 3 y 4 semanas a esta misma hora. "
        "Y los **promedios** (media de la misma hora en 4 semanas, nivel de la última semana conocida) "
        "resumen varias semanas en un número, para que una semana rara no arrastre al modelo. La última "
        "columna es el target: las llamadas que de verdad entraron, lo que el modelo tiene que acertar."
    )
    ver_codigo(features.variables_calendario, features.armar_features)


def pagina_validacion(ruta, simulados):
    st.header("5. Validación temporal")
    st.info(
        "**La idea:** medir el error sin mentirme. Si mezclo las filas al azar (el k-fold de "
        "siempre), el modelo entrena con datos del futuro y el error da más lindo de lo que va "
        "a ser en la realidad. El walk-forward repite lo que haría en producción: entrenar con "
        "el pasado, predecir la semana siguiente — y el experimento de abajo mide cuánto miente "
        "cada forma."
    )
    st.write(
        "En el curso usamos k-fold: se mezclan las filas al azar y se separan en grupos. Con series eso "
        "hace trampa, porque para predecir un martes el modelo entrena con el lunes y el miércoles de esa "
        "misma semana, y con semanas posteriores. En la realidad el futuro nunca está disponible."
    )
    st.write(
        "El **walk-forward** respeta el orden: entreno con todo lo anterior a una semana, pruebo en esa "
        "semana, después agrego esa semana al entrenamiento y pruebo en la siguiente."
    )

    semanas = st.slider("Semanas de prueba", 3, 12, 6)
    tabla = cargar_tabla(ruta)
    filas = []
    for i, (ent, prueba) in enumerate(validacion.particiones(len(tabla), semanas, HORIZONTE)):
        nombre = f"Ronda {i + 1}"
        filas.append({"Ronda": nombre, "Inicio": tabla.index[ent[0]], "Fin": tabla.index[ent[-1]],
                      "Tipo": "Entrenamiento"})
        filas.append({"Ronda": nombre, "Inicio": tabla.index[prueba[0]], "Fin": tabla.index[prueba[-1]],
                      "Tipo": "Prueba"})
    fig = px.timeline(pd.DataFrame(filas), x_start="Inicio", x_end="Fin", y="Ronda", color="Tipo",
                      color_discrete_map={"Entrenamiento": "#1f77b4", "Prueba": "#ff7f0e"})
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=120 + 35 * semanas, legend_title="")
    st.plotly_chart(fig)
    st.caption("Cada fila es una ronda: con el período azul se entrena y en el naranja se mide el error. "
               "La prueba siempre está después del entrenamiento, y el entrenamiento crece de ronda en "
               "ronda, igual que en la vida real: cada semana que pasa hay más historia disponible.")

    st.subheader("Experimento: ¿cuál de las dos estima mejor el error real?")
    st.write(
        "Separo las últimas 4 semanas como un 'futuro' que no toco. Con el resto estimo el error de Gradient "
        "Boosting de las dos formas, sobre las mismas 6 semanas. Después entreno con todo el pasado, predigo "
        "el futuro y comparo."
    )
    res = correr_comparacion(ruta)
    df = pd.DataFrame({"Forma": list(res), "MAE": list(res.values())})
    fig = px.bar(df, x="Forma", y="MAE", text_auto=".2f", color="Forma",
                 color_discrete_sequence=["#ff7f0e", "#1f77b4", "#2b2b2b"])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Error medio (llamadas por hora)")
    st.plotly_chart(fig)

    kf, wf, real = res["K-fold al azar"], res["Walk-forward"], res["Error real en el futuro"]
    st.write(
        f"Sobre **las mismas semanas**, el k-fold estima un error de **{kf:.2f}** y el walk-forward de "
        f"**{wf:.2f}**: el k-fold promete un {100 * (1 - kf / wf):.0f}% menos solo por haber entrenado "
        f"con filas vecinas de las que después evalúa. El error real en las 4 semanas de futuro fue "
        f"**{real:.2f}**."
    )
    # cual de las dos estimaciones quedo mas cerca del error real depende de los datos
    # (si el volumen cambia, el error absoluto se mueve con el), asi que el texto lo decide aca
    if abs(wf - real) <= abs(kf - real):
        st.write(
            "El walk-forward fue el que mejor anticipó ese error, porque repite exactamente cómo se "
            "va a usar el modelo: entrenar con el pasado, predecir la semana que sigue. Con k-fold "
            "habría prometido un error que después no se cumple."
        )
    else:
        st.write(
            "Ojo al interpretar la comparación con el futuro: acá el número del k-fold quedó más cerca "
            "del error real, pero es una casualidad del nivel de la serie — el volumen de llamadas de "
            "esas semanas cambió, y el error absoluto se mueve con él. La estimación metodológicamente "
            "válida sigue siendo la del walk-forward: es la única que nunca entrena con datos "
            "posteriores a los que predice, que es la situación real de producción."
        )
    ver_codigo(validacion.particiones, validacion.backtest, validacion.comparar_validaciones)


def pagina_modelos(ruta, simulados):
    st.header("6. Comparación de modelos")
    st.info(
        "**La idea:** cinco modelos compiten en igualdad de condiciones (misma tabla, misma "
        "validación) y gana el de menor error en horario de atención. El piso de la competencia "
        "es el naive: si un modelo no le gana a «repetir la semana pasada», sobra. Abajo también "
        "se ve **dónde** se equivoca el ganador y **qué variables** usa para decidir."
    )
    st.markdown(
        "- **Naive estacional**: predice lo mismo que la misma hora de la semana pasada. Es la vara mínima.\n"
        "- **Holt-Winters**: suavizado exponencial con nivel, tendencia y estacionalidad semanal. El modelo "
        "clásico de referencia: no usa features, trabaja directo sobre la serie.\n"
        "- **Ridge**: regresión lineal con regularización. La hora de la semana va en one-hot.\n"
        "- **Random Forest**: promedio de muchos árboles entrenados con muestras bootstrap (bagging).\n"
        "- **Gradient Boosting**: árboles que se agregan de a uno, cada uno corrigiendo el error de los anteriores."
    )
    st.write(
        "Los cinco compiten en igualdad de condiciones: misma tabla de features, mismo walk-forward y "
        "mismas semanas de prueba. En cada ronda, cada modelo se entrena desde cero con el pasado y "
        "predice la semana siguiente; lo único que cambia entre ellos es cómo aprenden la relación "
        "entre las features y las llamadas."
    )

    semanas = st.slider("Semanas de prueba (walk-forward)", 4, 12, 8)
    res = correr_backtest(ruta, semanas)
    res = res.join(cargar_tabla(ruta)[["hora_semana"]])
    abiertas = solo_horas_abiertas(res, ruta)
    nombres = [c for c in COLORES if c in res.columns and c != "real"]
    tm = metricas.tabla_metricas(abiertas, nombres)

    c1, c2 = st.columns([3, 2])
    c1.dataframe(tm, hide_index=True, column_config={
        "MAE": st.column_config.NumberColumn(format="%.2f"),
        "RMSE": st.column_config.NumberColumn(format="%.2f"),
        "WAPE (%)": st.column_config.NumberColumn(format="%.1f"),
        "Mejora vs naive (%)": st.column_config.NumberColumn(format="%.1f"),
    })
    fig = px.bar(tm, x="Modelo", y="MAE", color="Modelo", color_discrete_map=COLORES, text_auto=".2f")
    fig.update_layout(showlegend=False, height=300, xaxis_title="")
    c2.plotly_chart(fig)
    with st.expander("Qué significa cada métrica"):
        st.markdown(
            "- **MAE**: promedio de |real − predicho|, en llamadas por hora. Es la principal porque se "
            "lee directo: «me equivoco en tantas llamadas por hora».\n"
            "- **RMSE**: parecido, pero eleva el error al cuadrado antes de promediar, así que castiga "
            "mucho más las pifiadas grandes.\n"
            "- **WAPE**: el error total como porcentaje de las llamadas totales. Uso este y no el MAPE "
            "porque hay horas con 0 llamadas y el MAPE dividiría entre cero.\n"
            "- **Mejora vs naive**: cuánto más chico es el MAE que el del naive estacional. Un modelo "
            "que no le gana al naive no aporta nada: con repetir la semana pasada alcanzaba."
        )

    mejor = tm.iloc[0]
    st.write(
        f"El mejor fue **{mejor['Modelo']}**, con un error medio de {mejor['MAE']:.1f} llamadas por hora "
        f"({mejor['Mejora vs naive (%)']:.0f}% menos que el naive estacional). Las métricas se calculan solo en "
        f"horario de atención: predecir 0 de madrugada es trivial y achicaría el error artificialmente."
    )

    st.subheader("Predicción vs realidad")
    c1, c2 = st.columns([1, 3])
    sem = c1.selectbox("Semana de prueba", list(range(1, semanas + 1)), index=semanas - 1)
    elegidos = c2.multiselect("Modelos", nombres, default=["Naive estacional", mejor["Modelo"]])
    una = res[res["semana"] == sem]
    fig = go.Figure()
    fig.add_scatter(x=una.index, y=una["real"], name="Real", line=dict(color=COLORES["real"], width=3))
    for m in elegidos:
        fig.add_scatter(x=una.index, y=una[m], name=m, line=dict(color=COLORES[m]))
    fig.update_layout(height=400, yaxis_title="Llamadas", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)
    st.caption("La línea negra es lo que pasó de verdad; las demás, lo que cada modelo había predicho "
               "para esa semana sin haberla visto. Cambiá de semana o agregá modelos para comparar.")

    st.subheader("¿Dónde se equivoca?")
    c1, c2 = st.columns(2)
    abiertas = abiertas.assign(hora=abiertas.index.hour)
    por_hora = pd.DataFrame({m: abiertas.groupby("hora").apply(lambda g: metricas.mae(g["real"], g[m]))
                             for m in nombres})
    fig = px.line(por_hora, color_discrete_map=COLORES, markers=True,
                  labels={"value": "MAE", "hora": "Hora", "variable": "Modelo"}, title="Error según la hora")
    c1.plotly_chart(fig)
    por_sem = pd.DataFrame({m: abiertas.groupby("semana").apply(lambda g: metricas.mae(g["real"], g[m]))
                            for m in nombres})
    fig = px.bar(por_sem, barmode="group", color_discrete_map=COLORES,
                 labels={"value": "MAE", "semana": "Semana de prueba", "variable": "Modelo"},
                 title="Error según la semana")
    c2.plotly_chart(fig)
    st.write(
        "El error absoluto es más grande en las horas pico: con más llamadas hay más variabilidad "
        "(en un conteo tipo Poisson la varianza crece con la media). Si una semana tiene un error mucho más alto "
        "que las demás, conviene mirarla: suele ser un feriado o un evento que el modelo no podía saber."
    )

    st.subheader("¿Qué variables usa el modelo?")
    imp = correr_importancia(ruta)
    fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                 labels={"x": "Cuánto empeora el MAE si mezclo esa variable", "y": ""})
    fig.update_layout(height=420)
    st.plotly_chart(fig)
    st.caption("Importancia por permutación (Gradient Boosting, últimas 4 semanas): se desordena una columna "
               "por vez y se mide cuánto aumenta el error. Cuanto más larga la barra, más depende el "
               "modelo de esa variable; una barra en cero significa que el modelo la ignora.")
    ver_codigo(modelos.NaiveEstacional, modelos.HoltWinters, modelos.ridge, modelos.gradient_boosting,
               modelos.crear_modelos)


def pagina_incertidumbre(ruta, simulados):
    st.header("7. Incertidumbre: regresión cuantil")
    st.info(
        "**La idea:** el pronóstico normal dice «lo más probable», pero para poner gente "
        "necesito un «por las dudas». El cuantil 0,8 es un techo que la realidad solo supera "
        "2 de cada 10 veces: dimensiono contra ese techo y no contra el promedio, porque "
        "quedarme corto (gente esperando y cortando) cuesta más caro que pasarme."
    )
    st.write(
        "El modelo normal predice el valor esperado. Pero para dimensionar no da lo mismo equivocarse para "
        "arriba que para abajo: si faltan operadores la gente espera y corta, si sobran se pierde algo de plata. "
        "Por eso entreno además un modelo que predice un **cuantil**: el valor que las llamadas reales no "
        "deberían superar en un q% de los casos."
    )
    st.write("Se entrena igual que antes, pero minimizando la *pinball loss* en vez del error cuadrático:")
    st.latex(r"L_q(y, \hat{y}) = \begin{cases} q\,(y - \hat{y}) & \text{si } y \geq \hat{y} \\ "
             r"(1-q)\,(\hat{y} - y) & \text{si } y < \hat{y} \end{cases}")
    st.write("Con q = 0,8, quedarse corto pesa 4 veces más que pasarse, así que el modelo aprende a ir por arriba.")

    cuantiles = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    pred = correr_cuantiles(ruta, cuantiles)
    c1, c2 = st.columns([3, 1])
    q = c1.select_slider("Cuantil", options=list(cuantiles), value=0.8)
    sem = c2.selectbox("Semana", [1, 2, 3, 4], index=3, help="Las últimas 4 semanas, que el modelo no vio")

    una = pred.iloc[(sem - 1) * HORIZONTE: sem * HORIZONTE]
    alto = np.maximum(una[q], una[0.5])
    fig = go.Figure()
    fig.add_scatter(x=una.index, y=una[0.5], line=dict(color="#1f77b4", width=1), name="Mediana (q = 0,5)")
    fig.add_scatter(x=una.index, y=alto, fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
                    line=dict(color="#1f77b4", width=1, dash="dot"), name=f"Cuantil {q}")
    fig.add_scatter(x=una.index, y=una["real"], line=dict(color=COLORES["real"], width=2), name="Real")
    fig.update_layout(height=420, yaxis_title="Llamadas", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)
    st.caption("La banda va de la mediana al cuantil elegido. Si el cuantil está bien calibrado, la "
               "línea negra (lo real) tendría que quedar por debajo del borde superior más o menos en "
               "ese porcentaje de las horas: eso es exactamente lo que mide la cobertura de acá abajo.")
    if pred[0.5].max() < 1:
        st.info(
            "Con estos datos, la mediana (q = 0,5) se queda pegada en 0 y por eso su cobertura da "
            "tan baja: más de la mitad de las horas del año la central está cerrada, así que el "
            "cuantil 0,5 global de la serie es 0, el boosting arranca ahí y no logra despegar. Es "
            "una limitación conocida de los cuantiles bajos en series con muchas horas en cero. Los "
            "cuantiles altos —los que se usan para dimensionar— no la sufren, y su calibración se "
            "chequea acá abajo."
        )

    abiertas = solo_horas_abiertas(pred, ruta)
    c1, c2, c3 = st.columns(3)
    cob = metricas.cobertura(abiertas["real"], abiertas[q])
    c1.metric("Cobertura buscada", f"{100 * q:.0f} %")
    c2.metric("Cobertura obtenida", f"{100 * cob:.1f} %")
    c3.metric("Pinball loss", f"{metricas.pinball(abiertas['real'], abiertas[q], q):.2f}")
    st.write(
        "La cobertura es el porcentaje de horas en que las llamadas reales quedaron por debajo del cuantil "
        "predicho. Si el modelo está bien calibrado, tiene que dar parecido al cuantil pedido. "
        "Se mide en las últimas 4 semanas, que el modelo no vio."
    )
    if cob < q - 0.05:
        st.write(
            "Acá la cobertura quedó por debajo de lo pedido: en estas semanas hubo más llamadas que en el "
            "pasado, y un modelo de árboles no puede predecir valores más altos que los que vio al entrenar "
            "(no extrapola). Se podría corregir calibrando el cuantil con los errores de la validación, "
            "o modelando la tendencia aparte."
        )
    tabla_cob = pd.DataFrame({
        "Cuantil": cuantiles,
        "Cobertura obtenida (%)": [100 * metricas.cobertura(abiertas["real"], abiertas[x]) for x in cuantiles],
    })
    st.dataframe(tabla_cob, hide_index=True,
                 column_config={"Cobertura obtenida (%)": st.column_config.NumberColumn(format="%.1f")})
    ver_codigo(modelos.modelo_cuantil, metricas.pinball, metricas.cobertura)


def pagina_pronostico(ruta, simulados):
    st.header("8. Pronóstico y operadores")
    st.info(
        "**La idea:** acá se junta todo. El modelo que ganó el paso 6 pronostica la semana que "
        "viene, el cuantil del paso 7 pone el escenario alto, y la teoría de colas (Erlang C) "
        "convierte llamadas en personas. El mapa naranja de abajo es el entregable final: "
        "cuántos operadores en cada franja. Y al final, Erlang A hace el control de calidad: "
        "¿qué porcentaje de las llamadas atendería esa dotación?"
    )
    serie = cargar_serie(ruta)
    c1, c2 = st.columns(2)
    semanas = c1.radio("¿Con cuánta anticipación pronostico?", [1, 2, 3, 4], horizontal=True,
                       format_func=lambda s: f"{s} semana" if s == 1 else f"{s} semanas",
                       help="Los turnos se arman por semana, por eso el default es 1.")
    q = c2.select_slider("Escenario para dimensionar (cuantil)", options=[0.5, 0.7, 0.8, 0.9], value=0.8)
    h = semanas * HORIZONTE
    etiqueta = "1 semana" if semanas == 1 else f"{semanas} semanas"
    nombre = mejor_modelo(ruta)

    lags_disp = [c.replace("lag_", "") for c in cargar_tabla(ruta, h).columns if c.startswith("lag_")]
    c1, c2 = st.columns([2, 3])
    c1.write(
        f"El horizonte manda sobre las features: para predecir con {etiqueta} de anticipación solo "
        f"puedo usar lo que ya se conoce a esa distancia. De los lags semanales quedan disponibles: "
        f"**{', '.join(lags_disp)} horas** ({len(lags_disp)} de 4). Menos información, y más vieja: "
        f"en principio eso debería costar error, aunque cuánto cuesta depende de los datos — el "
        f"medidor de al lado lo mide de verdad en vez de suponerlo. El default es 1 semana porque es "
        f"lo que necesita la planificación de turnos."
    )
    # el backtest por horizonte es pesado (entrena 4 tablas distintas), asi que solo
    # corre si lo piden: si corriera solo, la pagina quedaria colgada un rato al entrar
    if c2.toggle("Medir el error según el horizonte", key="medir_horizonte",
                 help="Backtest del ganador y del naive a 1, 2, 3 y 4 semanas. "
                      "La primera vez tarda, después queda guardado."):
        err = error_por_horizonte(ruta, nombre)
        fig = px.line(err, markers=True, color_discrete_map=COLORES,
                      labels={"value": "MAE (llamadas por hora)", "variable": "Modelo"},
                      title="El precio de mirar más lejos")
        fig.update_layout(height=320, xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4]))
        c2.plotly_chart(fig)

    pron = correr_pronostico(ruta, q, nombre, h)

    historia = serie["llamadas"].iloc[-3 * HORIZONTE:]
    fig = go.Figure()
    fig.add_scatter(x=historia.index, y=historia.values, name="Real", line=dict(color=COLORES["real"]))
    fig.add_scatter(x=pron.index, y=pron["pronostico"], name=f"Pronóstico ({nombre})", line=dict(color="#d62728"))
    fig.add_scatter(x=pron.index, y=pron["alto"], name=f"Escenario alto (q = {q})", fill="tonexty",
                    fillcolor="rgba(214,39,40,0.15)", line=dict(color="#d62728", dash="dot", width=1))
    marcar_momento(fig, pron.index[0], "último dato", color="#555555")
    fig.update_layout(height=420, yaxis_title="Llamadas", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)
    st.write(
        f"La media la pronostica **{nombre}**, que fue el que ganó el backtest a una semana del paso 6 "
        f"(para horizontes más largos se reentrena ese mismo modelo con los lags válidos a esa distancia); "
        f"el escenario alto siempre es el Gradient Boosting con pérdida cuantil, que es el único de los "
        f"cinco que puede predecir un cuantil. El modelo se entrena con todos los datos hasta el "
        f"{serie.index.max().strftime('%d/%m/%Y')} y predice las {h} horas siguientes de una sola vez: "
        f"como solo usa lags de {h} horas o más, todo lo que necesita ya se conoce hoy (estrategia directa)."
    )

    st.subheader("De llamadas a operadores")
    st.write(
        "Esta parte no es machine learning: es teoría de colas. La fórmula de **Erlang C** dice cuántos "
        "operadores hacen falta para atender un porcentaje de llamadas en menos de cierto tiempo, "
        "sabiendo cuántas llamadas entran y cuánto dura cada una."
    )
    st.write(
        "Los controles de abajo son decisiones del negocio, no del modelo: qué nivel de servicio "
        "prometer (por ejemplo, atender el 80 % de las llamadas en menos de 20 segundos) y qué parte "
        "del turno un operador no está disponible (descansos, baños, capacitaciones). Probá subir el "
        "nivel de servicio y mirá el mapa: el costo de prometer mucho crece rápido."
    )
    aht_def = int(serie["aht_seg"].iloc[-4 * HORIZONTE:].median())
    c1, c2, c3, c4 = st.columns(4)
    aht = c1.number_input("Duración media (seg)", 30, 1200, aht_def, help="Mediana de las últimas 4 semanas")
    nivel = c2.slider("Nivel de servicio", 0.5, 0.95, 0.8, 0.05)
    segundos = c3.number_input("Atendidas en menos de (seg)", 5, 120, 20)
    perdida = c4.slider("Descansos y ausencias", 0.0, 0.4, 0.15, 0.05,
                        help="Porcentaje del turno en que el operador no está atendiendo")

    def operadores(llamadas):
        # erlang da los que tienen que estar atendiendo; divido por (1 - perdida) para
        # cubrir descansos y ausencias (lo que en los call centers llaman shrinkage)
        n = erlang.operadores_necesarios(llamadas, aht, nivel, segundos)
        return math.ceil(n / (1 - perdida))

    # en las horas en que la central esta cerrada no pongo a nadie
    abiertas = metricas.horas_de_atencion(cargar_tabla(ruta))
    abierto = (pron.index.day_of_week * 24 + pron.index.hour).isin(abiertas)
    pron["con_pronostico"] = np.where(abierto, pron["pronostico"].apply(operadores), 0)
    pron["con_escenario_alto"] = np.where(abierto, pron["alto"].apply(operadores), 0)
    # para el chequeo con erlang A preciso los que estan atendiendo de verdad
    # (sin el colchon de descansos), que es lo que devuelve erlang C
    atendiendo = pron["alto"].apply(lambda x: erlang.operadores_necesarios(x, aht, nivel, segundos))
    pron["atendiendo"] = np.where(abierto, atendiendo, 0).astype(int)

    tabla_op = pron["con_escenario_alto"].replace(0, np.nan).to_frame("op")
    # una fila por fecha (no por dia de la semana): asi el mapa funciona igual
    # cuando el pronostico es de mas de una semana
    tabla_op["fecha"] = tabla_op.index.normalize()
    tabla_op["hora"] = tabla_op.index.hour
    mapa = tabla_op.pivot_table(index="fecha", columns="hora", values="op")
    dias_futuro = pd.date_range(pron.index[0].normalize(), pron.index[-1].normalize(), freq="D")
    mapa = mapa.reindex(dias_futuro)  # los dias cerrados quedan como fila vacia, no desaparecen
    mapa.index = [DIAS[d.day_of_week] + " " + d.strftime("%d/%m") for d in dias_futuro]
    fig = px.imshow(mapa, text_auto=True, color_continuous_scale="Oranges", aspect="auto",
                    labels=dict(x="Hora", y="", color="Operadores"),
                    title=f"Operadores necesarios por franja (escenario q = {q})")
    fig.update_layout(height=max(380, 26 * len(mapa) + 130))
    st.plotly_chart(fig)
    st.caption("Este mapa es el entregable final: cada celda dice cuántas personas tienen que estar "
               "conectadas en esa franja del período pronosticado, ya con descansos y ausencias "
               "incluidos. Las horas en que la central está cerrada quedan vacías.")

    h_media = pron["con_pronostico"].sum()
    h_alto = pron["con_escenario_alto"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Horas-operador con el pronóstico medio", int(h_media))
    c2.metric(f"Horas-operador con el escenario q = {q}", int(h_alto))
    c3.metric("Diferencia", f"+{int(h_alto - h_media)}", f"{100 * (h_alto / max(h_media, 1) - 1):.1f} %")
    st.write(
        "La diferencia es el 'seguro' que se paga para no quedarse corto en las horas más inciertas. "
        "Cuánto conviene pagar es una decisión del call center, no del modelo."
    )

    st.subheader("El chequeo con Erlang A: ¿qué nivel de atención logro?")
    st.write(
        "Erlang C dimensiona con una hipótesis irreal a propósito: supone que nadie corta, así que "
        "tiende a pedir gente de más — ideal para decidir la dotación. **Erlang A** es el mismo "
        "modelo de colas pero con abandonos: cada persona en cola corta si su espera supera su "
        "paciencia. Acá lo uso al revés que a C: dada la dotación del mapa, ¿qué porcentaje de las "
        "llamadas voy a atender?"
    )
    pac = st.number_input("Paciencia media antes de cortar (seg)", 10, 600,
                          int(round(paciencia_media(ruta))),
                          help="Estimada con la espera promedio de las llamadas perdidas del histórico")

    def nivel_atencion(demanda):
        # perdidas esperadas hora a hora segun erlang A, sobre la demanda total del periodo
        perdidas_est = sum(d * erlang.erlang_a(d, aht, pac, n)
                           for d, n in zip(demanda, pron["atendiendo"]) if d > 0)
        return 100 * (1 - perdidas_est / max(demanda.sum(), 1e-9))

    historico = 100 * (1 - serie["abandonadas"].sum() / serie["llamadas"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Nivel de atención esperado", f"{nivel_atencion(pron['pronostico']):.1f} %",
              help="Con la dotación del mapa y la demanda del pronóstico medio")
    c2.metric(f"Si la demanda viene alta (q = {q})", f"{nivel_atencion(pron['alto']):.1f} %")
    c3.metric("Histórico real", f"{historico:.1f} %")
    st.write(
        "La lectura es una comparación: si el nivel esperado supera al histórico, la dotación "
        "propuesta atiende mejor de lo que se venía atendiendo, incluso si la demanda sale alta. "
        "Si diera peor, el nivel de servicio o el cuantil elegidos quedaron demasiado flojos."
    )
    st.caption("Supuestos de Erlang A: llegadas Poisson, duración y paciencia exponenciales. La "
               "paciencia sale de la espera de las perdidas, que es una medida censurada: solo veo "
               "cuánto esperó el que cortó, no cuánto más habría aguantado.")

    salida = pron.copy()
    salida.index.name = "fecha_hora"
    st.download_button("Descargar pronóstico (CSV)", salida.round(1).to_csv(),
                       f"pronostico_{etiqueta.replace(' ', '_')}.csv", "text/csv")
    ver_codigo(erlang.erlang_c, erlang.nivel_servicio, erlang.operadores_necesarios, erlang.erlang_a)


# ---------- menu ----------

# el logo de la universidad arriba de todo; si falta el archivo no rompo nada
_logo = Path("informe/logo_ucu.png")
if _logo.exists():
    st.sidebar.image(str(_logo), width=180)
st.sidebar.title("Pronóstico de llamadas")
st.sidebar.caption("Trabajo final · Machine Learning · UCU")

fuentes = {"Simulados": datos.RUTA_SIMULADA}
if Path(datos.RUTA_REAL).exists():
    fuentes = {"Reales": datos.RUTA_REAL, **fuentes}
fuente = st.sidebar.radio("Datos", list(fuentes))
ruta = fuentes[fuente]
simulados = ruta == datos.RUTA_SIMULADA

# el export real tambien se puede subir desde aca, sin tocar carpetas a mano.
# acepto csv o excel, pero lo que guardo siempre es un csv en data/raw/,
# la misma ruta que usa main.py: asi el resto del pipeline lee un solo formato
subido = st.sidebar.file_uploader("Subir export real (.csv o .xlsx)", type=["csv", "xlsx"])
if subido is not None:
    contenido = subido.getvalue()
    destino = Path(datos.RUTA_REAL)
    # el uploader devuelve el archivo en cada rerun: me acuerdo (en la sesion)
    # de cual ya procese para no convertir y limpiar el cache al pedo
    marca = hash((subido.name, contenido))
    if st.session_state.get("subida_procesada") != marca or not destino.exists():
        tmp = destino.with_name("_subida_tmp.csv")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            if subido.name.lower().endswith(".xlsx"):
                # el excel lo convierto a csv. los personales como texto,
                # si no el telefono pierde el 0 de adelante
                tipos = {c: str for c in datos.COLUMNAS_PERSONALES}
                pd.read_excel(io.BytesIO(contenido), dtype=tipos).to_csv(tmp, index=False)
            else:
                tmp.write_bytes(contenido)
            # valido antes de pisar nada: tiene que ser alguno de los dos
            # formatos que entiendo, y el parser tiene que poder leerlo entero
            encabezado = pd.read_csv(tmp, nrows=0).columns
            if datos.es_agregado(encabezado):
                datos.cargar_agregado(tmp)
            else:
                faltan = [c for c in datos.COLUMNAS.values() if c not in encabezado]
                if faltan:
                    raise ValueError(
                        "no lo reconozco ni como el reporte por hora de la central "
                        "ni como un export por llamada (faltan: " + ", ".join(faltan) + ")"
                    )
                datos.cargar_export(tmp)
        except Exception as e:
            # no marco la subida como procesada: el error se sigue mostrando
            # hasta que saquen o cambien el archivo
            tmp.unlink(missing_ok=True)
            st.sidebar.error(f"No pude usar ese archivo: {e}")
        else:
            tmp.replace(destino)
            st.session_state["subida_procesada"] = marca
            # los caches estan atados a la ruta del archivo, no a su contenido,
            # asi que al reemplazarlo hay que tirarlos para que se recalcule todo
            st.cache_data.clear()
            st.toast("Export cargado: ahora estás viendo los datos reales.")
            st.rerun()

PAGINAS = {
    "Inicio": inicio,
    "1. Datos crudos": pagina_crudos,
    "2. Anonimización": pagina_anonimizacion,
    "3. Exploración": pagina_exploracion,
    "4. De serie a tabla": pagina_serie_a_tabla,
    "5. Validación": pagina_validacion,
    "6. Modelos": pagina_modelos,
    "7. Incertidumbre": pagina_incertidumbre,
    "8. Pronóstico y operadores": pagina_pronostico,
}
st.sidebar.divider()
pagina = st.sidebar.radio("Recorrido", list(PAGINAS))
PAGINAS[pagina](ruta, simulados)
