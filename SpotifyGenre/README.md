# Spotify Genre Classifier

Sistema de clasificación de géneros musicales desarrollado con scikit-learn, FastAPI y ngrok.  
El modelo predice el género de una canción a partir de sus características acústicas extraídas del dataset de Spotify.

---

## Integrantes

- Mariana Montoya Sepúlveda
- María José Arcila
- Sebastián López

---

## Estructura del proyecto

```
spotify-genre-project/
├── app/
│   ├── __init__.py
│   └── main.py               # API con FastAPI
├── models/
│   └── gb_pipeline.pkl       # Modelo entrenado (Pipeline)
├── notebooks/
│   └── 01_modelo.ipynb       # Notebook principal de análisis y modelado
├── spotify_songs.csv         # Dataset original
└── README.md
```

---

## Configuración del entorno

### Requisitos

- Python 3.11 (recomendado para compatibilidad con todas las librerías)
- pip actualizado

### 1. Crear el entorno virtual

```bash
python -m venv .venv_clean
```

### 2. Activar el entorno

En Windows:

```bash
.venv_clean\Scripts\activate
```

En Linux/Mac:

```bash
source .venv_clean/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install numpy==1.26.4 pandas==2.1.4 scikit-learn==1.4.2 scipy==1.11.4 matplotlib==3.7.5 seaborn==0.13.2 joblib==1.3.2 fastapi uvicorn pyngrok
```

> **Nota:** PyCaret solo soporta Python 3.9, 3.10 y 3.11. Si usas Python 3.12, la instalación de PyCaret fallará con un
`RuntimeError`. Ver sección [PyCaret](#-pycaret).

---

## Ejecutar la API localmente

Desde la raíz del proyecto:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abre en el navegador:

```
http://127.0.0.1:8000/docs
```

---

## Exponer la API públicamente con ngrok

Ejecuta en el notebook o en una celda aparte:

```python
from pyngrok import ngrok

ngrok.set_auth_token("TU_AUTHTOKEN")
public_url = ngrok.connect(8000)
print(public_url)
```

La URL generada tipo `https://xxxx.ngrok-free.dev` permite consumir la API desde cualquier lugar.

> El authtoken lo obtienes desde tu cuenta
> en [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).

---

## Descripción del dataset

El dataset `spotify_songs.csv` contiene canciones de Spotify con características acústicas y metadatos.  
La variable objetivo es `playlist_genre`, que clasifica cada canción en una de las siguientes categorías:

| Género | Descripción            |
|--------|------------------------|
| edm    | Electronic Dance Music |
| latin  | Música latina          |
| pop    | Pop                    |
| r&b    | Rhythm and Blues       |
| rap    | Rap / Hip-Hop          |
| rock   | Rock                   |

### Variables independientes utilizadas

```
trackpopularity, danceability, energy, key, loudness,
mode, speechiness, acousticness, instrumentalness,
liveness, valence, tempo, durationms
```

Se eliminaron columnas de texto e identificadores como `track_id`, `track_name`, `track_artist`, entre otras.

---

## Data Cleaning

Se aplicaron los siguientes procesos de limpieza:

- Eliminación de columnas no numéricas irrelevantes para el modelado.
- Revisión y eliminación de valores nulos.
- Verificación de tipos de datos por columna.
- Revisión de estadísticas descriptivas para detectar valores atípicos evidentes.

---

## Análisis Exploratorio

- Descripción estadística con `.describe()`.
- Análisis de correlación entre variables numéricas con mapa de calor.
- Identificación de variables con baja o alta correlación respecto al target.
- Distribución de clases del target para detectar desbalance.

---

## Ingeniería de características

No se aplicó ingeniería de características compleja.  
Se realizó selección de variables mediante análisis de correlación y se excluyeron columnas de identificación y texto.  
El escalado de datos se incluyó dentro del `Pipeline` del modelo final.

---

## Modelado

Se evaluaron múltiples algoritmos de clasificación con scikit-learn.  
El modelo con mejor desempeño fue **GradientBoostingClassifier**, seleccionado con base en las métricas de accuracy,
precision, recall y F1-score en el conjunto de prueba.

### Métricas del mejor modelo

| Métrica              | Valor |
|----------------------|-------|
| Accuracy             | ~0.59 |
| Precision (weighted) | ~0.58 |
| Recall (weighted)    | ~0.59 |
| F1-score (weighted)  | ~0.58 |

---

## Pipeline (scikit-learn)

Se utilizó la clase `Pipeline` de scikit-learn para encadenar el preprocesamiento (escalado con `StandardScaler`) y el
modelo de clasificación en una sola estructura reproducible.

**Ventajas:**

- Evita inconsistencias entre entrenamiento y predicción.
- Facilita el despliegue del modelo en producción.
- Compatible con `cross_val_score` y `GridSearchCV`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(random_state=42))
])
```

El modelo fue guardado con `joblib`:

```python
import joblib
joblib.dump(pipeline, "models/gb_pipeline.pkl")
```

---

## Ensamble

Se implementó un `VotingClassifier` con tres estimadores:

- `LogisticRegression` (dentro de un Pipeline con StandardScaler)
- `RandomForestClassifier`
- `DecisionTreeClassifier`

**Resultado:** El ensamble no superó al mejor modelo individual. La regresión logística presentó advertencias de
convergencia y el accuracy global del ensamble fue de ~0.44, inferior al obtenido con GradientBoosting.

**Conclusión:** Se priorizó el modelo individual por su mejor desempeño y mayor estabilidad.

---

## FastAPI

Se implementó una API REST con FastAPI que expone el modelo entrenado para realizar predicciones en tiempo real.

### Endpoint principal

**POST** `/predict`

**Body de entrada (JSON):**

```json
{
  "trackpopularity": 65.0,
  "danceability": 0.72,
  "energy": 0.85,
  "key": 5.0,
  "loudness": -5.3,
  "mode": 1.0,
  "speechiness": 0.08,
  "acousticness": 0.12,
  "instrumentalness": 0.0,
  "liveness": 0.11,
  "valence": 0.65,
  "tempo": 128.0,
  "durationms": 210000.0
}
```

**Respuesta:**

```json
{
  "playlistgenre": "pop"
}
```

### Documentación interactiva

Con la API corriendo, abre en el navegador:

```
http://127.0.0.1:8000/docs
```

---

## PyCaret

PyCaret fue investigado como herramienta de AutoML para automatizar la comparación de modelos.  
**No fue posible usarlo en este proyecto** porque la librería soporta oficialmente Python 3.9, 3.10 y 3.11, mientras que
el entorno de desarrollo utilizó Python 3.12.

Al intentar importar PyCaret, se generó el siguiente error:

```
RuntimeError: Pycaret only supports python 3.9, 3.10, 3.11. Please DOWNGRADE your Python version.
```

Por esta razón, la comparación de modelos se realizó con scikit-learn, que ofreció compatibilidad completa y mayor
control sobre el proceso.

**Referencias:**

- PyCaret. (2024). *Installation*. https://pycaret.readthedocs.io/en/stable/installation.html
- PyCaret. (2024). *PyPI*. https://pypi.org/project/pycaret/

---

## Estado del proyecto

| Componente                                | Estado                          |
|-------------------------------------------|---------------------------------|
| Data Cleaning                             | ✅ Completado                    |
| Análisis exploratorio                     | ✅ Completado                    |
| Análisis de correlación                   | ✅ Completado                    |
| Modelado con scikit-learn                 | ✅ Completado                    |
| Matriz de confusión                       | ✅ Completado                    |
| Métricas de efectividad                   | ✅ Completado                    |
| Pipeline                                  | ✅ Completado                    |
| Guardado del modelo (joblib)              | ✅ Completado                    |
| FastAPI (local)                           | ✅ Completado                    |
| FastAPI (público con ngrok)               | ✅ Completado                    |
| Ensamble (VotingClassifier)               | ✅ Completado                    |
| PyCaret                                   | ⚠️ Incompatible con Python 3.12 |
| Ingeniería de características documentada | ⚠️ Pendiente de ampliar         |
| Despliegue permanente en la nube          | ⚠️ Pendiente                    |

---

## ⚠️ Pendientes

- **PyCaret:** Para usarlo correctamente se necesita un entorno con Python 3.10 o 3.11.
- **Despliegue permanente:** La URL de ngrok es temporal. Para acceso permanente se recomienda desplegar en Render,
  Railway o Koyeb.
- **Ingeniería de características:** Se puede explorar la creación de nuevas variables derivadas y técnicas de selección
  más avanzadas como PCA o SelectKBest.
- **Ajuste de hiperparámetros:** No se realizó búsqueda sistemática con GridSearchCV o RandomizedSearchCV sobre el
  modelo final.
- **Balanceo de clases:** Se puede explorar SMOTE u otras técnicas si el desbalance afecta el rendimiento.

---

## 📚 Referencias

- scikit-learn. (s. f.). *Pipeline*. https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- FastAPI. (s. f.). *Tutorial*. https://fastapi.tiangolo.com/tutorial/
- PyCaret. (2024). *Installation*. https://pycaret.readthedocs.io/en/stable/installation.html
- PyCaret. (2024). *PyPI*. https://pypi.org/project/pycaret/
- ngrok. (2026). *Using ngrok with FastAPI*. https://ngrok.com/docs/using-ngrok-with/fastAPI
