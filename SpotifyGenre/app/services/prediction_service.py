from pathlib import Path

import joblib
import pandas as pd
from app.core.logger import get_logger
from app.schemas.song_schema import SongInput

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "gb_pipeline.pkl"

try:
    logger.info(f"Cargando modelo desde {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    model = None


def make_prediction(data: SongInput) -> str:
    if model is None:
        logger.error("Se intentó predecir pero el modelo no está cargado.")
        raise ValueError("El modelo de Machine Learning no está disponible.")

    logger.info("Procesando datos para nueva predicción...")
    df_input = pd.DataFrame([data.model_dump()])

    pred = model.predict(df_input)[0]
    logger.info(f"Predicción exitosa: {pred}")

    return pred
