from fastapi import APIRouter, HTTPException
from app.schemas.song_schema import SongInput, PredictionOutput
from app.services.prediction_service import make_prediction
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=PredictionOutput)
def predict_genre(data: SongInput):
    logger.info("Recibida petición POST en /predict")
    try:
        resultado = make_prediction(data)
        return PredictionOutput(playlistgenre=str(resultado))
    except Exception as e:
        logger.error(f"Fallo en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno procesando la predicción")