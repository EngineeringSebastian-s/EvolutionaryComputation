from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Spotify Genre Classifier",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR.parent / "models" / "gb_pipeline.pkl")


class SongInput(BaseModel):
    trackpopularity: float
    danceability: float
    energy: float
    key: float
    loudness: float
    mode: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    durationms: float


@app.get("/")
def home():
    pass


@app.get("/")
def home():
    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict(data: SongInput):
    df_input = pd.DataFrame([data.model_dump()])
    pred = model.predict(df_input)[0]
    return {"playlistgenre": pred}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
