import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import time
import os
import io
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    layout="wide", page_title="Dashboard Interactivo - Fórmula 1", page_icon="🏎️"
)


# ========== FUNCIONES EXPORT ==========
def exportar_grafico(fig, nombre_archivo, texto_boton):
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    return st.download_button(
        texto_boton, buffer.getvalue(), nombre_archivo, "image/png"
    )


def exportar_csv(df, nombre_archivo, texto_boton):
    return st.download_button(
        texto_boton, df.to_csv(index=False), nombre_archivo, "text/csv"
    )


def exportar_excel(feedback_paths, nombre_salida):
    buffer_excel = io.BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        for nombre, ruta in feedback_paths.items():
            if os.path.exists(ruta):
                pd.read_csv(ruta).to_excel(writer, sheet_name=nombre, index=False)
    buffer_excel.seek(0)
    return buffer_excel


# ========== FUNCIONES DE CARGA Y PROCESAMIENTO ==========
@st.cache_data(ttl=86400, show_spinner="Cargando resultados desde la API...")
def cargar_datos_api(temporada):
    registros = []
    offset = 0
    while True:
        url = f"https://api.jolpi.ca/ergast/f1/{temporada}/results.json?limit=100&offset={offset}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            races = data["MRData"]["RaceTable"]["Races"]
            if not races:
                break
            for carrera in races:
                circuito = carrera["Circuit"]["circuitName"]
                ronda = carrera["round"]
                fecha = carrera["date"]
                pais = carrera["Circuit"]["Location"]["country"]
                for result in carrera["Results"]:
                    registros.append(
                        {
                            "Temporada": temporada,
                            "Ronda": int(ronda),
                            "Fecha": fecha,
                            "Circuito": circuito,
                            "Pais": pais,
                            "Piloto": f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                            "Escudería": result["Constructor"]["name"],
                            "Posición": int(result["position"]),
                            "Puntos": float(result["points"]),
                            "Status": result["status"],
                        }
                    )
            offset += 100
        except Exception as e:
            st.error(f"Error al cargar resultados: {e}")
            break
    return pd.DataFrame(registros)


@st.cache_data(ttl=86400)
def cargar_posiciones_clasificacion(temporada):
    posiciones = []
    for ronda in range(1, 25):
        url = f"https://api.jolpi.ca/ergast/f1/{temporada}/{ronda}/qualifying.json"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                continue
            data = response.json()
            resultados = data["MRData"]["RaceTable"]["Races"]
            if not resultados:
                continue
            for result in resultados[0]["QualifyingResults"]:
                piloto = (
                    f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
                )
                posiciones.append(
                    {
                        "Temporada": temporada,
                        "Ronda": int(ronda),
                        "Piloto": piloto,
                        "PosicionClasificacion": int(result["position"]),
                    }
                )
        except Exception:
            continue
    return pd.DataFrame(posiciones)


@st.cache_data(ttl=86400)
def obtener_lat_lon_circuitos():
    url = "https://api.jolpi.ca/ergast/f1/circuits.json?limit=100"
    r = requests.get(url)
    datos = r.json()["MRData"]["CircuitTable"]["Circuits"]
    return pd.DataFrame(
        [
            {
                "Circuito": c["circuitName"],
                "Lat": float(c["Location"]["lat"]),
                "Lon": float(c["Location"]["long"]),
            }
            for c in datos
        ]
    )


@st.cache_data(ttl=86400)
def obtener_clima_para_carreras(df_resultados):
    clima_por_carrera = {}
    for _, fila in (
        df_resultados[["Circuito", "Fecha", "Lat", "Lon"]].drop_duplicates().iterrows()
    ):
        clave = (fila["Circuito"], fila["Fecha"])
        lat, lon = fila["Lat"], fila["Lon"]
        fecha = fila["Fecha"]
        if clave in clima_por_carrera:
            continue
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": fecha,
            "end_date": fecha,
            "daily": "precipitation_sum",
            "timezone": "UTC",
        }
        try:
            resp = requests.get(url, params=params)
            datos = resp.json()
            lluvia_mm = datos["daily"]["precipitation_sum"][0]
            clima = "Lluvia" if lluvia_mm > 0.1 else "Seco"
        except Exception:
            clima = "Desconocido"
        clima_por_carrera[clave] = clima
        time.sleep(0.8)
    df_resultados["Clima"] = df_resultados.apply(
        lambda row: clima_por_carrera.get(
            (row["Circuito"], row["Fecha"]), "Desconocido"
        ),
        axis=1,
    )
    return df_resultados


def obtener_temporadas_disponibles():
    anio_actual = datetime.now().year
    return list(range(2003, anio_actual + 1))


# ========== SIDEBAR FILTROS ==========
with st.sidebar:
    st.title("🏎️ Dashboard F1 - Filtros globales")
    temporadas = obtener_temporadas_disponibles()
    temporada = st.selectbox("Temporada", temporadas, index=len(temporadas) - 1)
    with st.spinner("Cargando datos..."):
        if f"df_{temporada}" not in st.session_state:
            df = cargar_datos_api(temporada)
            df_clasif = cargar_posiciones_clasificacion(temporada)
            df_coords = obtener_lat_lon_circuitos()
            if all(
                col in df.columns for col in ["Temporada", "Ronda", "Piloto"]
            ) and all(
                col in df_clasif.columns for col in ["Temporada", "Ronda", "Piloto"]
            ):
                df = df.merge(
                    df_clasif, on=["Temporada", "Ronda", "Piloto"], how="left"
                )
            else:
                st.error(
                    "❌ No se pudo hacer el merge: faltan columnas en los datos cargados."
                )
                st.stop()
            df = df.merge(df_coords, on="Circuito", how="left")
            df = obtener_clima_para_carreras(df)
            st.session_state[f"df_{temporada}"] = df
        else:
            df = st.session_state[f"df_{temporada}"]

    paises_disponibles = sorted(df["Pais"].dropna().unique())
    pais_seleccionado = st.selectbox("Filtrar por país", ["Todos"] + paises_disponibles)
    climas_disponibles = sorted(df["Clima"].dropna().unique())
    clima_seleccionado = st.multiselect(
        "Filtrar por clima", climas_disponibles, default=climas_disponibles
    )
    pilotos_disponibles = sorted(df["Piloto"].unique())
    piloto = st.selectbox("Filtrar por piloto", ["Todos"] + pilotos_disponibles)
    escuderias_disponibles = sorted(df["Escudería"].unique())
    escuderia = st.selectbox(
        "Filtrar por escudería", ["Todos"] + escuderias_disponibles
    )

    df_filtrado = df.copy()
    if pais_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Pais"] == pais_seleccionado]
    if clima_seleccionado:
        df_filtrado = df_filtrado[df_filtrado["Clima"].isin(clima_seleccionado)]
    if piloto != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Piloto"] == piloto]
    if escuderia != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Escudería"] == escuderia]
    st.markdown("---")
    exportar_csv(df_filtrado, "datos_filtrados.csv", "Descargar CSV filtrado")

# ========== COLUMNAS DERIVADAS ==========
df_filtrado["Victoria"] = df_filtrado["Posición"] == 1
df_filtrado["Abandono"] = ~df_filtrado["Status"].str.contains(
    "Finished", case=False, na=True
)

tabs = st.tabs(
    [
        "Mapa",
        "Rankings",
        "Comparaciones",
        "Predicción",
        "Métricas",
        "Correlación",
        "Radar",
        "Feedback",
        "Conclusiones",
    ]
)

# ========== TAB: MAPA ==========
with tabs[0]:
    st.title("🗺️ Mapa interactivo de circuitos")
    df_map = (
        df_filtrado[["Lat", "Lon", "Circuito", "Pais", "Fecha"]]
        .dropna()
        .rename(columns={"Lat": "latitude", "Lon": "longitude"})
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=df_map["latitude"].mean() if not df_map.empty else 0,
                longitude=df_map["longitude"].mean() if not df_map.empty else 0,
                zoom=1,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position="[longitude, latitude]",
                    get_radius=50000,
                    get_fill_color="[180, 0, 200, 140]",
                    pickable=True,
                )
            ],
            tooltip={
                "html": "<b>Circuito:</b> {Circuito}<br/>"
                "<b>País:</b> {Pais}<br/>"
                "<b>Fecha:</b> {Fecha}",
                "style": {"backgroundColor": "steelblue", "color": "white"},
            },
        )
    )

# ========== TAB: RANKINGS ==========
with tabs[1]:
    st.title("🏁 Rankings")
    ranking_pilotos = (
        df_filtrado.groupby("Piloto")["Puntos"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranking_escuderias = (
        df_filtrado.groupby("Escudería")["Puntos"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pilotos")
        st.dataframe(ranking_pilotos)
        exportar_csv(
            ranking_pilotos, "ranking_pilotos.csv", "Descargar ranking de pilotos"
        )
    with col2:
        st.subheader("Escuderías")
        st.dataframe(ranking_escuderias)
        exportar_csv(
            ranking_escuderias,
            "ranking_escuderias.csv",
            "Descargar ranking de escuderías",
        )

# ========== TAB: COMPARACIONES ==========
with tabs[2]:
    st.title("🆚 Comparaciones entre pilotos y escuderías")
    pilotos_sel = st.multiselect(
        "Seleccioná pilotos",
        sorted(df_filtrado["Piloto"].unique()),
        default=sorted(df_filtrado["Piloto"].unique())[:2],
    )
    if pilotos_sel:
        promedios = (
            df_filtrado[df_filtrado["Piloto"].isin(pilotos_sel)]
            .groupby("Piloto")["Puntos"]
            .mean()
            .reset_index()
        )
        fig1 = px.bar(
            promedios,
            x="Piloto",
            y="Puntos",
            title="Puntos promedio por piloto",
            text="Puntos",
        )
        st.plotly_chart(fig1)
        df_comp = df_filtrado[df_filtrado["Piloto"].isin(pilotos_sel)].copy()
        df_comp["Ronda"] = pd.to_numeric(df_comp["Ronda"])
        df_comp["Acumulado"] = df_comp.groupby("Piloto")["Puntos"].cumsum()
        fig2 = px.line(
            df_comp,
            x="Ronda",
            y="Acumulado",
            color="Piloto",
            title="Evolución de puntos acumulados",
        )
        st.plotly_chart(fig2)
        clima_comparado = (
            df_filtrado[df_filtrado["Piloto"].isin(pilotos_sel)]
            .groupby(["Piloto", "Clima"])["Puntos"]
            .mean()
            .reset_index()
        )
        fig3 = px.bar(
            clima_comparado,
            x="Clima",
            y="Puntos",
            color="Piloto",
            barmode="group",
            title="Rendimiento por clima",
        )
        st.plotly_chart(fig3)

    escuderias_sel = st.multiselect(
        "Seleccioná escuderías",
        sorted(df_filtrado["Escudería"].unique()),
        default=sorted(df_filtrado["Escudería"].unique())[:2],
    )
    if escuderias_sel:
        promedios_escu = (
            df_filtrado[df_filtrado["Escudería"].isin(escuderias_sel)]
            .groupby("Escudería")["Puntos"]
            .mean()
            .reset_index()
        )
        fig4 = px.bar(
            promedios_escu,
            x="Escudería",
            y="Puntos",
            title="Puntos promedio por escudería",
            text="Puntos",
        )
        st.plotly_chart(fig4)
        df_escu = df_filtrado[df_filtrado["Escudería"].isin(escuderias_sel)].copy()
        df_escu["Ronda"] = pd.to_numeric(df_escu["Ronda"])
        df_escu["Acumulado"] = df_escu.groupby("Escudería")["Puntos"].cumsum()
        fig5 = px.line(
            df_escu,
            x="Ronda",
            y="Acumulado",
            color="Escudería",
            title="Evolución de puntos acumulados por escudería",
        )
        st.plotly_chart(fig5)
        clima_escu = (
            df_filtrado[df_filtrado["Escudería"].isin(escuderias_sel)]
            .groupby(["Escudería", "Clima"])["Puntos"]
            .mean()
            .reset_index()
        )
        fig6 = px.bar(
            clima_escu,
            x="Clima",
            y="Puntos",
            color="Escudería",
            barmode="group",
            title="Rendimiento por clima de escuderías",
        )
        st.plotly_chart(fig6)

# ========== TAB: PREDICCIÓN ==========
with tabs[3]:
    st.title("🤖 Predicción de posición final")
    df_modelo = df_filtrado[
        ["PosicionClasificacion", "Posición", "Clima", "Escudería"]
    ].dropna()
    if not df_modelo.empty:
        X = df_modelo[["PosicionClasificacion", "Clima", "Escudería"]].copy()
        X["Clima"] = X["Clima"].astype(str)
        X["Escudería"] = X["Escudería"].astype(str)
        y = df_modelo["Posición"]
        pre = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                    ["Clima", "Escudería"],
                )
            ],
            remainder="passthrough",
        )
        modelo = Pipeline(
            [
                ("prep", pre),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(
            f"📏 Error absoluto medio del modelo: **{int(round(mae))} posiciones**"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            clasif = st.slider("Posición de clasificación", 1, 20, 5)
        with col2:
            clima_in = st.selectbox("Clima de la carrera", ["Seco", "Lluvia"])
        with col3:
            escu_in = st.selectbox(
                "Escudería", sorted(df_filtrado["Escudería"].unique())
            )

        if st.button("Predecir posición final"):
            df_input = pd.DataFrame(
                [
                    {
                        "PosicionClasificacion": clasif,
                        "Clima": clima_in,
                        "Escudería": escu_in,
                    }
                ]
            )
            df_input["Clima"] = df_input["Clima"].astype(str)
            df_input["Escudería"] = df_input["Escudería"].astype(str)
            pred = modelo.predict(df_input)[0]
            pos_entera = int(round(pred))
            pos_entera = max(1, min(20, pos_entera))  # Asegura que esté entre 1 y 20
            st.success(
                f"🔮 La posición final estimada para este piloto es: **{pos_entera}°**"
            )
    else:
        st.info(
            "No hay datos suficientes para entrenar el modelo con los filtros aplicados."
        )

# ========== TAB: MÉTRICAS ==========
with tabs[4]:
    st.title("📈 Métricas de desempeño")
    metricas = (
        df_filtrado.groupby("Piloto")
        .agg({"Victoria": "sum", "Posición": "mean", "Abandono": "mean"})
        .reset_index()
        .rename(
            columns={
                "Victoria": "Carreras Ganadas",
                "Posición": "Promedio de Posición",
                "Abandono": "% Abandono",
            }
        )
    )
    metricas["% Abandono"] = (metricas["% Abandono"] * 100).round(2)
    st.dataframe(
        metricas.sort_values(
            by=["Carreras Ganadas", "Promedio de Posición", "% Abandono"],
            ascending=[False, True, True],
        )
    )
    st.subheader("Escuderías")
    metricas_escu = (
        df_filtrado.groupby("Escudería")
        .agg({"Victoria": "sum", "Posición": "mean", "Abandono": "mean"})
        .reset_index()
        .rename(
            columns={
                "Victoria": "Carreras Ganadas",
                "Posición": "Promedio de Posición",
                "Abandono": "% Abandono",
            }
        )
    )
    metricas_escu["% Abandono"] = (metricas_escu["% Abandono"] * 100).round(2)
    st.dataframe(
        metricas_escu.sort_values(
            by=["Carreras Ganadas", "Promedio de Posición", "% Abandono"],
            ascending=[False, True, True],
        )
    )

# ========== TAB: CORRELACIÓN Y ANÁLISIS AVANZADO ==========
with tabs[5]:
    st.title("🔬 Correlaciones y Análisis de consistencia")
    st.subheader("Correlación entre posición de clasificación y posición final")
    corr = df_filtrado[["PosicionClasificacion", "Posición"]].corr().iloc[0, 1]
    st.write(f"Coeficiente de correlación: **{corr:.2f}**")
    fig_corr = px.scatter(
        df_filtrado, x="PosicionClasificacion", y="Posición"
    )  # SACÁ trendline='ols'
    st.plotly_chart(fig_corr)

    st.subheader("Boxplot de posición final por clima y piloto")
    pilotos_sel_box = st.multiselect(
        "Seleccioná pilotos para el boxplot",
        sorted(df_filtrado["Piloto"].unique()),
        default=sorted(df_filtrado["Piloto"].unique())[:5],
        key="boxplot_pilotos",
    )
    if pilotos_sel_box:
        df_box = df_filtrado[df_filtrado["Piloto"].isin(pilotos_sel_box)]
        fig_box = px.box(
            df_box,
            x="Piloto",
            y="Posición",
            color="Clima",
            points="all",
            title="Distribución de posiciones finales por clima y piloto",
        )
        st.plotly_chart(fig_box)

    st.subheader("Consistencia de pilotos (STD de posiciones finales, menor es mejor)")
    consistencia = (
        df_filtrado.groupby("Piloto")["Posición"]
        .std()
        .reset_index()
        .rename(columns={"Posición": "STD_Posición"})
    )
    consistencia = consistencia.sort_values("STD_Posición")
    st.dataframe(consistencia)

    st.subheader(
        "Consistencia de escuderías (STD de posiciones finales, menor es mejor)"
    )
    consistencia_escu = (
        df_filtrado.groupby("Escudería")["Posición"]
        .std()
        .reset_index()
        .rename(columns={"Posición": "STD_Posición"})
    )
    consistencia_escu = consistencia_escu.sort_values("STD_Posición")
    st.dataframe(consistencia_escu)
    fig_escu_std = px.bar(
        consistencia_escu,
        x="Escudería",
        y="STD_Posición",
        title="Consistencia de escuderías (menor es mejor)",
    )
    st.plotly_chart(fig_escu_std)

    st.subheader("Diferencia de rendimiento de pilotos entre seco y lluvia")
    # Solo consideramos pilotos que corrieron en ambos climas
    pivot_clima = df_filtrado.pivot_table(
        index="Piloto", columns="Clima", values="Puntos", aggfunc="mean"
    )
    if "Seco" in pivot_clima.columns and "Lluvia" in pivot_clima.columns:
        pivot_clima = pivot_clima[["Seco", "Lluvia"]].dropna()
        pivot_clima["Diferencia_Lluvia-Seco"] = (
            pivot_clima["Lluvia"] - pivot_clima["Seco"]
        )
        st.dataframe(pivot_clima.sort_values("Diferencia_Lluvia-Seco"))
        fig_diff = px.bar(
            pivot_clima.reset_index(),
            x="Piloto",
            y="Diferencia_Lluvia-Seco",
            title="Diferencia de puntos promedio por piloto: Lluvia - Seco",
            color="Diferencia_Lluvia-Seco",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig_diff)
    else:
        st.info(
            "No hay suficientes datos de ambos climas para comparar el rendimiento de los pilotos."
        )

# ========== TAB: RADAR ==========
with tabs[6]:
    st.title("🕹️ Radar de desempeño")

    radar_metricas = (
        df_filtrado.groupby("Piloto")
        .agg({"Puntos": "mean", "Victoria": "mean", "Abandono": "mean"})
        .reset_index()
        .rename(
            columns={
                "Puntos": "Promedio de Puntos",
                "Victoria": "Tasa de Victoria",
                "Abandono": "Tasa de Abandono",
            }
        )
    )

    pilotos_radar = st.multiselect(
        "Seleccioná pilotos para el radar",
        radar_metricas["Piloto"].unique(),
        default=list(radar_metricas["Piloto"].unique())[:3],
    )
    if pilotos_radar:
        df_radar = radar_metricas[radar_metricas["Piloto"].isin(pilotos_radar)]
        categories = ["Promedio de Puntos", "Tasa de Victoria", "Tasa de Abandono"]
        fig = go.Figure()
        for i, row in df_radar.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=[row[c] for c in categories],
                    theta=categories,
                    fill="toself",
                    name=row["Piloto"],
                )
            )
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig)

# ========== TAB: FEEDBACK ==========

with tabs[7]:
    # Definí las rutas al inicio del tab
    ruta_pilotos = "feedback_pilotos.csv"
    ruta_escuderias = "feedback_escuderias.csv"

    st.title("🗣️ Feedback")
    st.markdown("¿Qué tipo de visualización te resulta más útil para comparar pilotos?")
    preferencia_piloto = st.radio(
        "Visualización favorita para pilotos:",
        ["Gráfico", "Tabla", "Ambos", "Ninguno"],
        key="feedback_piloto",
        index=None,
    )

    if st.button("Enviar feedback de pilotos"):
        if preferencia_piloto is not None:
            respuesta_pilotos = pd.DataFrame(
                [
                    {
                        "Temporada": temporada,
                        "Fecha": pd.Timestamp.now(),
                        "Preferencia": preferencia_piloto,
                    }
                ]
            )
            respuesta_pilotos.to_csv(
                ruta_pilotos,
                mode="a",
                index=False,
                header=not os.path.exists(ruta_pilotos),
            )
            st.success(
                f"Gracias por tu respuesta sobre pilotos: **{preferencia_piloto}**"
            )
        else:
            st.warning("Por favor, seleccioná una opción antes de enviar el feedback.")

    st.markdown("---")
    st.markdown(
        "¿Qué tipo de visualización te resulta más útil para comparar escuderías?"
    )
    preferencia_escu = st.radio(
        "Visualización favorita para escuderías:",
        ["Gráfico", "Tabla", "Ambos", "Ninguno"],
        key="feedback_escu",
        index=None,
    )

    if st.button("Enviar feedback de escuderías"):
        if preferencia_escu is not None:
            respuesta_escuderias = pd.DataFrame(
                [
                    {
                        "Temporada": temporada,
                        "Fecha": pd.Timestamp.now(),
                        "Preferencia": preferencia_escu,
                    }
                ]
            )
            respuesta_escuderias.to_csv(
                ruta_escuderias,
                mode="a",
                index=False,
                header=not os.path.exists(ruta_escuderias),
            )
            st.success(
                f"Gracias por tu respuesta sobre escuderías: **{preferencia_escu}**"
            )
        else:
            st.warning("Por favor, seleccioná una opción antes de enviar el feedback.")

    st.markdown("---")
    st.markdown("### Descargar resumen de feedback")
    resumen_paths = {"Pilotos": ruta_pilotos, "Escuderías": ruta_escuderias}
    buffer_excel = exportar_excel(resumen_paths, "resumen_feedback.xlsx")
    st.download_button(
        "⬇️ Descargar resumen de feedback",
        buffer_excel,
        "resumen_feedback.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if st.button("Borrar feedback guardado"):
        try:
            if os.path.exists(ruta_pilotos):
                os.remove(ruta_pilotos)
            if os.path.exists(ruta_escuderias):
                os.remove(ruta_escuderias)
            st.success("Feedback borrado correctamente.")
        except Exception as e:
            st.error(f"Error al borrar el archivo: {e}")

# ========== TAB: CONCLUSIONES ==========
with tabs[8]:
    st.title("📜 Conclusiones automáticas (Hipótesis)")
    # H1: Visualización preferida
    if os.path.exists(ruta_pilotos):
        feedback = pd.read_csv(ruta_pilotos)
        conteo_h1 = feedback["Preferencia"].value_counts()
        if "Gráfico" in conteo_h1 and conteo_h1["Gráfico"] > conteo_h1.get("Tabla", 0):
            conclusion_h1 = "✅ La mayoría de los usuarios prefirió el gráfico interactivo sobre la tabla."
        else:
            conclusion_h1 = "❌ La tabla fue igual o más clara que el gráfico según el feedback recolectado."
    else:
        conclusion_h1 = "⚠️ No se pudo evaluar H1 por falta de datos de feedback."
    st.markdown(f"- **H1:** {conclusion_h1}")

    # H2: Correlación entre clasificación y resultado final
    corr = df_filtrado[["PosicionClasificacion", "Posición"]].corr().iloc[0, 1]
    if abs(corr) > 0.5:
        conclusion_h2 = f"✅ Fuerte correlación ({corr:.2f}) entre posición de clasificación y posición final."
    else:
        conclusion_h2 = f"❌ Correlación débil ({corr:.2f}) entre posición de clasificación y resultado final."
    st.markdown(f"- **H2:** {conclusion_h2}")

    # H3: Consistencia de escuderías (más y menos consistente)
    if "Posición" in df_filtrado.columns:
        escu_consist = df_filtrado.groupby("Escudería")["Posición"].std().sort_values()
        mejor = escu_consist.idxmin()
        peor = escu_consist.idxmax()
        conclusion_h3 = (
            f"Escudería más consistente: **{mejor}** (STD posición: {escu_consist.min():.2f})\n\n"
            f"Escudería menos consistente: **{peor}** (STD posición: {escu_consist.max():.2f})"
        )
    else:
        conclusion_h3 = "⚠️ No hay datos de posiciones finales para evaluar H3."
    st.markdown(f"- **H3:** {conclusion_h3}")

    # H4: Piloto más beneficiado/perjudicado por la lluvia
    # (usamos mismo pivot_clima que en tab de correlación, lo recalculamos para estar seguros)
    pivot_clima = df_filtrado.pivot_table(
        index="Piloto", columns="Clima", values="Puntos", aggfunc="mean"
    )
    if (
        "Seco" in pivot_clima.columns
        and "Lluvia" in pivot_clima.columns
        and not pivot_clima.empty
    ):
        pivot_clima = pivot_clima[["Seco", "Lluvia"]].dropna()
        mayor = pivot_clima["Diferencia_Lluvia-Seco"] = (
            pivot_clima["Lluvia"] - pivot_clima["Seco"]
        )
        mayor_piloto = pivot_clima["Diferencia_Lluvia-Seco"].idxmax()
        menor_piloto = pivot_clima["Diferencia_Lluvia-Seco"].idxmin()
        mejora = pivot_clima.loc[mayor_piloto, "Diferencia_Lluvia-Seco"]
        empeora = pivot_clima.loc[menor_piloto, "Diferencia_Lluvia-Seco"]
        conclusion_h4 = (
            f"Piloto que **más mejora** bajo lluvia: **{mayor_piloto}** (gana {mejora:.2f} puntos promedio respecto a seco).\n\n"
            f"Piloto que **más empeora** bajo lluvia: **{menor_piloto}** (pierde {abs(empeora):.2f} puntos promedio respecto a seco)."
        )
    else:
        conclusion_h4 = "⚠️ No hay suficientes datos para determinar qué piloto mejora o empeora bajo la lluvia."
    st.markdown(f"- **H4:** {conclusion_h4}")

st.markdown("---")
st.markdown(
    "Desarrollado por Emanuel Gallo y Facundo Olcese | Proyecto Integrador F1 | 2025"
)
