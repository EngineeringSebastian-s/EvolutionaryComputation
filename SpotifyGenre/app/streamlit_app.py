import os
import time

import requests
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Spotify Genre Predictor",
    page_icon="🎵",
    layout="wide"
)

# Diccionario para darle estilo a cada género
GENRE_STYLES = {
    "edm": {"emoji": "🎧", "color": "#1DB954", "desc": "Música Electrónica y Dance. ¡Sube los bajos!"},
    "latin": {"emoji": "💃", "color": "#FF4500", "desc": "Ritmos Latinos. ¡A bailar!"},
    "pop": {"emoji": "🎤", "color": "#FF1493", "desc": "Pop comercial. Pegadizo y popular."},
    "r&b": {"emoji": "🎷", "color": "#8A2BE2", "desc": "Rhythm and Blues. Suave y con mucho groove."},
    "rap": {"emoji": "🧢", "color": "#FFD700", "desc": "Rap & Hip-Hop. Barras y beats marcados."},
    "rock": {"emoji": "🎸", "color": "#8B0000", "desc": "Rock clásico o moderno. ¡Pura energía con guitarras!"}
}

st.title("🎵 Spotify Genre Classifier - Dashboard")
st.markdown(
    "Ajusta los parámetros musicales en el panel lateral y mira cómo el modelo predice el género en tiempo real.")

# --- PANEL LATERAL (SLIDERS) ---
st.sidebar.header("🎛️ Panel de Mezcla")

# Usamos valores por defecto basados en las medias de tu notebook
track_popularity = st.sidebar.slider("Popularidad", 0, 100, 50, help="0 a 100")
danceability = st.sidebar.slider("Bailabilidad (Danceability)", 0.0, 1.0, 0.65, 0.01)
energy = st.sidebar.slider("Energía (Energy)", 0.0, 1.0, 0.70, 0.01)
loudness = st.sidebar.slider("Volumen (Loudness dB)", -50.0, 5.0, -6.0, 0.1)
tempo = st.sidebar.slider("Tempo (BPM)", 50.0, 220.0, 120.0, 1.0)
valence = st.sidebar.slider("Positividad (Valence)", 0.0, 1.0, 0.50, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Detalles Acústicos")
acousticness = st.sidebar.slider("Acústico (Acousticness)", 0.0, 1.0, 0.15, 0.01)
instrumentalness = st.sidebar.slider("Instrumentalidad", 0.0, 1.0, 0.05, 0.01)
speechiness = st.sidebar.slider("Hablado (Speechiness)", 0.0, 1.0, 0.10, 0.01)
liveness = st.sidebar.slider("En Vivo (Liveness)", 0.0, 1.0, 0.15, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Estructura Musical")
key = st.sidebar.slider("Clave Musical (Key)", 0, 11, 5)
mode = st.sidebar.radio("Modo", options=[0, 1], format_func=lambda x: "Menor (0)" if x == 0 else "Mayor (1)", index=1)
duration_ms = st.sidebar.slider("Duración (ms)", 60000, 400000, 200000, 1000, help="200,000 ms = ~3.3 minutos")

# Preparar los datos para la API
data = {
    "track_popularity": float(track_popularity),
    "danceability": float(danceability),
    "energy": float(energy),
    "key": float(key),
    "loudness": float(loudness),
    "mode": float(mode),
    "speechiness": float(speechiness),
    "acousticness": float(acousticness),
    "instrumentalness": float(instrumentalness),
    "liveness": float(liveness),
    "valence": float(valence),
    "tempo": float(tempo),
    "duration_ms": float(duration_ms)
}

col1, col2 = st.columns([1, 1])

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

with col1:
    st.subheader("📡 Conectando con la IA...")

    # Animación de carga fluida
    with st.spinner('Analizando los ritmos y frecuencias...'):
        time.sleep(0.3)  # Pequeña pausa para que se aprecie la animación
        try:
            # Hacer petición a tu FastAPI
            response = requests.post(API_URL, json=data)

            if response.status_code == 200:
                prediction = response.json().get("playlistgenre", "").lower()
            else:
                prediction = "error"
                st.error(f"Error en la API: {response.text}")
        except requests.exceptions.ConnectionError:
            prediction = "offline"
            st.error("⚠️ No se pudo conectar con FastAPI. Asegúrate de que Uvicorn esté corriendo en el puerto 8000.")

with col2:
    if prediction not in ["error", "offline"]:
        style = GENRE_STYLES.get(prediction, {"emoji": "🎵", "color": "#FFFFFF", "desc": "Género analizado"})

        st.markdown(f"""
            <div style="background-color: {style['color']}20; padding: 30px; border-radius: 15px; border: 2px solid {style['color']}; text-align: center;">
                <h1 style="font-size: 4rem; margin: 0;">{style['emoji']}</h1>
                <h2 style="color: {style['color']}; text-transform: uppercase; letter-spacing: 2px;">{prediction}</h2>
                <p style="font-size: 1.2rem; font-style: italic;">{style['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Opcional: Mostrar una gráfica de araña o barras de las características principales
        st.write("### Perfil de la pista")
        st.progress(danceability, text=f"Bailabilidad: {int(danceability * 100)}%")
        st.progress(energy, text=f"Energía: {int(energy * 100)}%")
        st.progress(valence, text=f"Positividad: {int(valence * 100)}%")
