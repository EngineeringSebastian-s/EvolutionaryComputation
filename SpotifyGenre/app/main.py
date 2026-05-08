import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.core.logger import get_logger
from app.routers import predict_router

logger = get_logger(__name__)

app = FastAPI(
    title="Spotify Genre Classifier",
    description="API estructurada para predicción de géneros musicales",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Incluimos los routers
app.include_router(predict_router.router)


@app.on_event("startup")
def startup_event():
    logger.info("Iniciando aplicación Spotify Genre Classifier...")


@app.get("/", include_in_schema=False)
def home():
    logger.info("Redirigiendo a la documentación...")
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
