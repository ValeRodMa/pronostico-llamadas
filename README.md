# Pronóstico de llamadas de un call center

Trabajo final de Machine Learning (UCU). Tema: series temporales.

El objetivo es predecir cuántas llamadas van a entrar a un call center en cada hora de la semana siguiente, y usar ese pronóstico para estimar cuántos operadores hacen falta en cada franja.

## Cómo correrlo en local

Hace falta Python 3.10 o más nuevo. Descargar el proyecto (botón Code → Download ZIP, o `git clone`) y en la carpeta:

```
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador (http://localhost:8501). La app recorre todo el proceso paso a paso: datos crudos, anonimización, exploración, armado de features, validación, comparación de modelos, incertidumbre y pronóstico. Si no hay datos reales, la primera vez genera datos simulados.

Desde la terminal:

```
python main.py evaluar        # backtest walk-forward de todos los modelos
python main.py pronosticar    # pronóstico de la semana siguiente + operadores
python main.py simular        # regenera los datos simulados
pytest                        # tests
```

## Datos

Los datos reales salen del sistema de la central y se aceptan en dos formatos: el reporte "Datos de las Colas" (ya agregado, una fila por hora, sin datos personales) o un export con una fila por llamada (como el simulado). No se suben al repositorio: van en `data/raw/llamadas.csv`, que está en el `.gitignore`. También se pueden subir directo desde la app (menú de la izquierda), en `.csv` o Excel `.xlsx`: si es Excel, la app lo convierte a csv y lo guarda en esa misma ruta. El formato se detecta solo por las columnas (`src/datos.py`).

Antes de cualquier análisis se borran las columnas con datos personales (teléfono, id de cliente, operador) y se cuentan las llamadas por hora. Si las columnas del export tienen otros nombres, hay que ajustar el diccionario `COLUMNAS` en `src/datos.py`.

## Estructura

```
app.py              app de Streamlit con el recorrido paso a paso
main.py             el mismo pipeline desde la terminal
src/
  datos.py          carga, anonimización, agregación por hora y datos simulados
  features.py       calendario, feriados de Uruguay, lags y medias móviles
  modelos.py        naive estacional, Holt-Winters, Ridge, Random Forest, Gradient Boosting y cuantiles
  validacion.py     backtest walk-forward y comparación con k-fold
  metricas.py       MAE, RMSE, WAPE, pinball loss y cobertura
  erlang.py         Erlang C (dimensionar operadores) y Erlang A (chequear el nivel de atención)
  exploracion.py    autocorrelación y mapa de calor
tests/
  test_features.py  verifica que ninguna feature use datos del futuro
  test_datos.py     detección de formato y carga del reporte por hora
  test_erlang.py
informe/
  informe.docx      el informe del trabajo: marco teórico, metodología y resultados
  logo_ucu.png      logo para la carátula
```

## Decisiones principales

- Horizonte de una semana (168 h), porque los turnos se planifican por semana. El modelo predice la semana entera de una vez, así que solo usa lags de 168 horas o más. La app permite estirarlo a 2, 3 o 4 semanas y mide con el mismo walk-forward cuánto cambia el error a cada distancia.
- Validación walk-forward (`TimeSeriesSplit`) en vez de k-fold, para no entrenar con datos posteriores a los que se predicen.
- Holt-Winters como referencia clásica en la comparación, con estacionalidad aditiva (de noche hay horas con 0 llamadas, la multiplicativa necesita valores positivos) y período 168 (el patrón semanal ya contiene al diario).
- La demanda son todas las llamadas que entraron (atendidas + abandonadas), no solo las atendidas.
- Las métricas se calculan solo en horario de atención.
- El pronóstico final no usa un modelo fijo: usa el que ganó el backtest, porque el ganador depende de los datos (con los simulados, que tienen tendencia lineal, gana Ridge; con los reales ganan los árboles).
- Además del modelo para la media, un Gradient Boosting con pérdida cuantil para tener un escenario alto, que es el que se usa para dimensionar.
- Erlang C decide la dotación (supone que nadie corta: conservador) y Erlang A la verifica: con esa dotación y la paciencia estimada del histórico, predice el nivel de atención esperado.
