# Usamos una imagen oficial y ligera de Python
FROM python:3.11-slim

# Evitamos que Python genere archivos .pyc y forzamos logs directos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos las dependencias del sistema necesarias para compilar algunas librerías
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiamos primero solo los requerimientos para aprovechar la caché de Docker
COPY requirements.txt .

# Instalamos las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código de la API y el modelo al contenedor
COPY ./app ./app
COPY ./models ./models

# Exponemos el puerto interno que usará FastAPI (Uvicorn)
EXPOSE 8000 8051

# Comando para iniciar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]