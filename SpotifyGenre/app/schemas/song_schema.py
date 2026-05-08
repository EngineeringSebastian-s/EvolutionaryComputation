from pydantic import BaseModel

class SongInput(BaseModel):
    track_popularity: float
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
    duration_ms: float

class PredictionOutput(BaseModel):
    playlistgenre: str