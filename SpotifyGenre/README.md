# Spotify Genre Classifier

Sistema de clasificacion de generos musicales desarrollado con scikit-learn, FastAPI y Docker.
El modelo predice el genero de una cancion a partir de sus caracteristicas acusticas extraidas del dataset de Spotify.

> **Actividad 03 - Aplicacion de modelos de ML en Python y prototipado usando Orange (15%)**
> Materia: ING01216 - Aprendizaje de Maquina y Computacion Evolutiva
> Docente: Jeison Alejandro Zapata Pulgarin

---

## Integrantes

- Maria Jose Arcila Cano
- Sebastian Lopez Osorio
- Mariana Montoya Sepulveda

---

## Despliegue publico

| Servicio                            | URL                          |
|-------------------------------------|------------------------------|
| Dashboard (Streamlit)               | http://72.60.68.22:8014      |
| API REST (FastAPI)                  | http://72.60.68.22:8013      |
| Documentacion interactiva (Swagger) | http://72.60.68.22:8013/docs |

> El servicio esta desplegado en un servidor publico mediante Docker y Docker Compose, accesible desde cualquier
> ubicacion en internet.

---

## Estructura del proyecto

```
spotify-genre-project/
├── app/
│   ├── __init__.py
│   ├── main.py               # API con FastAPI
│   └── streamlit_app.py      # Dashboard con Streamlit
├── models/
│   └── gb_pipeline.pkl       # Modelo entrenado (Pipeline)
├── notebooks/
│   └── 01_modelo.ipynb       # Notebook principal de analisis y modelado
├── spotify_songs.csv         # Dataset original
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Configuracion del entorno local

### Requisitos

- Python 3.11 (recomendado para compatibilidad con todas las librerias)
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
pip install -r requirements.txt
```

> **Nota:** PyCaret solo soporta Python 3.9, 3.10 y 3.11. Si usas Python 3.12, la instalacion fallara con un
> RuntimeError.

---

## Ejecutar la API localmente

Desde la raiz del proyecto:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abre en el navegador:

```
http://127.0.0.1:8000/docs
```

---

## Ejecutar con Docker

### Requisitos

- Docker instalado
- Docker Compose instalado

### Levantar todos los servicios

Desde la carpeta del proyecto:

```bash
docker compose up --build
```

| Servicio                  | Puerto local | Descripcion                    |
|---------------------------|--------------|--------------------------------|
| `spotify-genre-api`       | `8013`       | API REST con FastAPI           |
| `spotify-genre-dashboard` | `8014`       | Dashboard visual con Streamlit |

### Acceder a los servicios

```
http://localhost:8013       -> API REST
http://localhost:8013/docs  -> Swagger UI (documentacion interactiva)
http://localhost:8014       -> Dashboard Streamlit
```

### Detener los servicios

```bash
docker compose down
```

---

---

# Informe - Actividad 03

## Punto A - Prototipado usando Orange

Orange es una herramienta de prototipado visual para machine learning que permite construir flujos de trabajo mediante
componentes graficos interconectados (widgets), sin necesidad de escribir codigo. Cada widget representa una operacion:
cargar datos, preprocesar, entrenar un modelo, evaluar resultados o visualizar.

### Flujo de trabajo implementado en Orange

El prototipado se realizo con el siguiente flujo:

```
File (spotify_songs.csv)
    -> Select Columns (seleccion de variables independientes y target)
    -> Data Sampler (80% entrenamiento / 20% prueba)
    -> [Logistic Regression / Random Forest / Gradient Boosting / Decision Tree]
    -> Test and Score (evaluacion con accuracy, F1, AUC)
    -> Confusion Matrix (visualizacion de errores por clase)
```

### Resultados obtenidos en Orange

Orange permitio comparar rapidamente multiples modelos sobre el mismo conjunto de datos. Los resultados de accuracy por
modelo fueron consistentes con los obtenidos luego en Python, confirmando que GradientBoosting fue el mejor algoritmo
para este dataset.

| Modelo              | Accuracy (Orange) |
|---------------------|-------------------|
| Gradient Boosting   | ~0.58             |
| Random Forest       | ~0.55             |
| Decision Tree       | ~0.38             |
| Logistic Regression | ~0.36             |

### Diferencias entre Orange y Python

| Aspecto                       | Orange                    | Python (scikit-learn)         |
|-------------------------------|---------------------------|-------------------------------|
| Curva de aprendizaje          | Baja (visual, sin codigo) | Media (requiere programacion) |
| Control sobre hiperparametros | Limitado                  | Total                         |
| Reproducibilidad              | Baja (flujo grafico)      | Alta (codigo versionado)      |
| Integracion con despliegue    | No disponible             | Si (FastAPI, Docker)          |
| Velocidad de prototipado      | Alta                      | Media                         |
| Profundidad de analisis       | Basica                    | Completa                      |

Orange fue util como herramienta exploratoria para identificar rapidamente los modelos mas prometedores antes de
implementarlos en Python con mayor precision y control.

---

## Punto B - Aplicacion de modelos de ML en Python

---

### 1. Data Cleaning

Se identificaron y aplicaron los siguientes aspectos de limpieza sobre el dataset `spotify_songs.csv`:

- **Eliminacion de columnas no numericas:** Se removieron columnas de texto e identificadores (`track_id`, `track_name`,
  `track_album_id`, `track_album_name`, `track_album_release_date`, `playlist_name`, `playlist_id`, `playlist_subgenre`,
  `track_artist`) que no aportan informacion cuantitativa al modelo.
- **Revision de valores nulos:** Se identificaron y eliminaron filas con valores nulos en cualquiera de las variables
  restantes.
- **Verificacion de tipos de datos:** Se confirmo que todas las variables independientes son numericas (`float64` o
  `int64`).
- **Revision de estadisticas descriptivas:** Se utilizo `.describe()` para detectar rangos anomalos o valores extremos
  en variables como `loudness`, `tempo` y `durationms`.

---

### 2. Descripcion del dataset y analisis de correlacion

El dataset contiene registros de canciones de Spotify con caracteristicas acusticas extraidas automaticamente por la
plataforma.

**Variable objetivo (dependiente):** `playlist_genre`

| Genero | Descripcion            |
|--------|------------------------|
| edm    | Electronic Dance Music |
| latin  | Musica latina          |
| pop    | Pop                    |
| r&b    | Rhythm and Blues       |
| rap    | Rap / Hip-Hop          |
| rock   | Rock                   |

**Variables independientes seleccionadas:**

| Variable           | Descripcion                       |
|--------------------|-----------------------------------|
| `trackpopularity`  | Popularidad de la cancion (0-100) |
| `danceability`     | Que tan bailable es (0.0-1.0)     |
| `energy`           | Intensidad percibida (0.0-1.0)    |
| `key`              | Tonalidad musical (0-11)          |
| `loudness`         | Volumen promedio en dB            |
| `mode`             | Modalidad mayor/menor (0 o 1)     |
| `speechiness`      | Presencia de palabras habladas    |
| `acousticness`     | Probabilidad de ser acustica      |
| `instrumentalness` | Ausencia de voz                   |
| `liveness`         | Probabilidad de grabacion en vivo |
| `valence`          | Positividad musical (0.0-1.0)     |
| `tempo`            | Tempo en BPM                      |
| `durationms`       | Duracion en milisegundos          |

**Analisis de correlacion:**
El mapa de calor generado con `seaborn` mostro que ninguna variable tiene una correlacion lineal fuerte con el genero
musical, lo cual es esperado dado que la variable objetivo es categorica. Variables como `energy`, `acousticness` y
`danceability` presentaron mayor varianza entre generos y fueron mas relevantes para el modelo. Se identifico
correlacion negativa entre `energy` y `acousticness` (-0.73), lo que indica que las canciones mas energicas tienden a
ser menos acusticas. Variables como `key` y `mode` presentaron correlaciones muy bajas, pero se retuvieron por su
potencial aporte en modelos no lineales.

**Conjuntos de entrenamiento y prueba:**

- Entrenamiento: 80% de los datos
- Prueba: 20% de los datos
- Division realizada con `train_test_split(random_state=42, stratify=y)` para mantener la proporcion de clases.

**Conclusiones sobre el dataset:**

El dataset de Spotify es rico en variables acusticas continuas, pero presenta una limitacion estructural: los generos
musicales no tienen fronteras acusticas rigidas. Generos como `pop`, `r&b` y `latin` comparten rangos similares en
variables como `danceability` y `energy`, lo que hace que cualquier modelo de clasificacion enfrente solapamiento real
entre clases. La distribucion de clases esta relativamente equilibrada (aproximadamente 5.000 canciones por genero), por
lo que no se requirio balanceo artificial. El dataset tiene 13 variables numericas utiles tras la limpieza, lo que
representa una dimensionalidad manejable sin necesidad de reduccion.

---

### 3. Ingenieria de caracteristicas

**No fue necesario aplicar ingenieria de caracteristicas compleja.**

- Las variables disponibles en el dataset ya son representaciones acusticas directamente interpretables y procesadas por
  la API de Spotify.
- Se descarto la creacion de variables derivadas por la naturaleza ya normalizada del dataset.
- El unico proceso aplicado fue el escalado de variables con `StandardScaler`, incluido dentro del `Pipeline`.
- Se evaluo PCA, pero dado que no hay alta dimensionalidad (13 variables), no se justifico su aplicacion.

---

### 4. Modelos evaluados

**GradientBoostingClassifier (mejor modelo):**

```python
gb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=2,
        random_state=42
    ))
])
```

**RandomForestClassifier:**

```python
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])
```

GradientBoosting fue seleccionado tras comparar multiples algoritmos evaluando accuracy y F1-score sobre el conjunto de
prueba.

---

### 5. Matriz de confusion

La matriz de confusion se genero sobre el conjunto de prueba con las 6 clases de genero:

```
Clases: ['edm', 'latin', 'pop', 'r&b', 'rap', 'rock']
```

La diagonal de la matriz concentra los valores mas altos, indicando que el modelo clasifica correctamente la mayoria de
las instancias. Los generos con mayor confusion entre si fueron `pop` y `r&b`, lo cual es coherente con su similitud
acustica. El genero `rap` fue el mas facilmente separable por su alta `speechiness`.

> La imagen de la matriz de confusion se encuentra en el notebook `notebooks/01_modelo.ipynb`.

---

### 6. Metricas de efectividad

| Metrica              | GradientBoosting | RandomForest |
|----------------------|------------------|--------------|
| Accuracy             | ~0.59            | ~0.56        |
| Precision (weighted) | ~0.58            | ~0.56        |
| Recall (weighted)    | ~0.59            | ~0.56        |
| F1-score (weighted)  | ~0.58            | ~0.56        |

Las metricas fueron calculadas con `classification_report` de scikit-learn sobre el conjunto de prueba (20% de los
datos).

---

### 7. Conclusiones del modelado

- GradientBoosting fue el modelo con mejor desempeno general, con un accuracy de ~0.59 sobre datos de prueba, superando
  a RandomForest (~0.56) y a los demas algoritmos evaluados.
- Los generos musicales no tienen fronteras acusticas perfectamente definidas, lo que establece un techo de desempeno
  inherente al problema.
- La variable `acousticness` resulto ser la mas discriminante para separar `rock` y `edm` del resto, mientras que
  `speechiness` fue clave para identificar `rap`.
- El uso de `n_estimators=200` y `max_depth=4` en GradientBoosting permitio capturar patrones mas complejos sin
  sobreajuste excesivo.
- **Metodo seleccionado para implementacion:** GradientBoostingClassifier dentro de un Pipeline con StandardScaler.

---

## Punto C - FastAPI

Se implemento una API REST con FastAPI que expone el modelo entrenado para realizar predicciones en tiempo real desde
cualquier ubicacion en internet.

### Que es FastAPI

FastAPI es un framework moderno de Python para construir APIs REST de alto rendimiento, basado en anotaciones de tipo y
compatible con el estandar OpenAPI. Genera documentacion interactiva automatica con Swagger UI, lo que facilita el
consumo y la prueba del servicio sin necesidad de clientes externos (Ramirez, 2018).

### Implementacion

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

### Acceso publico

| Recurso               | URL                          |
|-----------------------|------------------------------|
| API base              | http://72.60.68.22:8013      |
| Documentacion Swagger | http://72.60.68.22:8013/docs |
| Dashboard visual      | http://72.60.68.22:8014      |

---

## Punto D - Pipeline (scikit-learn)

### Para que sirve Pipeline

`Pipeline` de scikit-learn permite encadenar multiples pasos de procesamiento y modelado en un unico objeto. Cada paso
recibe la salida del anterior, garantizando que la misma transformacion aplicada en entrenamiento se aplique exactamente
igual en prediccion (Buitinck et al., 2013).

**Ventajas principales:**

- Evita *data leakage*: el escalado se ajusta solo con datos de entrenamiento.
- Simplifica el despliegue: se serializa un solo objeto `.pkl` que incluye preprocesamiento y modelo.
- Compatible con `cross_val_score`, `GridSearchCV` y `joblib`.

### Implementacion

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

gb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=2,
        random_state=42
    ))
])

gb_pipeline.fit(X_train, y_train)
joblib.dump(gb_pipeline, "models/gb_pipeline.pkl")
```

---

## Punto E - PyCaret

### Que es PyCaret

PyCaret es una libreria de AutoML de bajo codigo para Python que automatiza la comparacion, ajuste y evaluacion de
multiples modelos de machine learning con pocas lineas de codigo (Ali, 2020).

### Resultado en el proyecto

**No fue posible usar PyCaret** porque la libreria soporta oficialmente Python 3.9, 3.10 y 3.11. El entorno de
desarrollo utilizo Python 3.12, lo que genero el siguiente error:

```
RuntimeError: Pycaret only supports python 3.9, 3.10, 3.11.
Please DOWNGRADE your Python version.
```

La comparacion de modelos se realizo manualmente con scikit-learn, que ofrece compatibilidad completa con Python 3.12.

**Que hubiera aportado PyCaret:**

- Comparacion automatica de mas de 20 modelos con `compare_models()`.
- Generacion automatica de metricas, graficas y matrices de confusion.
- Ajuste de hiperparametros con `tune_model()`.
- Exportacion directa del mejor modelo con `save_model()`.

---

## Punto F - Ensamble

### Implementacion

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000, solver="lbfgs"))
])

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

dt = DecisionTreeClassifier(random_state=42)

ensamble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("dt", dt)
    ],
    voting="hard"
)

ensamble.fit(X_train, y_train)
y_ens = ensamble.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_ens))
print("Precision:", precision_score(y_test, y_ens, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test, y_ens, average="weighted", zero_division=0))
print("F1:", f1_score(y_test, y_ens, average="weighted", zero_division=0))
print(confusion_matrix(y_test, y_ens))
print(classification_report(y_test, y_ens, zero_division=0))
```

### Comparacion de resultados

| Modelo                                   | Accuracy | F1-score (weighted) |
|------------------------------------------|----------|---------------------|
| GradientBoostingClassifier (Pipeline)    | ~0.59    | ~0.58               |
| RandomForestClassifier (Pipeline)        | ~0.56    | ~0.56               |
| VotingClassifier (ensamble LR + RF + DT) | ~0.44    | ~0.43               |

### Conclusiones del ensamble

- El `VotingClassifier` no supero al mejor modelo individual. Su accuracy fue aproximadamente 15 puntos porcentuales
  inferior al de GradientBoosting.
- La regresion logistica (`solver="lbfgs"`, `max_iter=3000`) es un modelo lineal en un problema que no es linealmente
  separable. Aunque se incremento `max_iter` para garantizar convergencia, el modelo aporto predicciones de menor
  calidad que los arboles, arrastrando el resultado del ensamble hacia abajo.
- El `DecisionTreeClassifier` sin restriccion de profundidad tiende al sobreajuste sobre datos de entrenamiento, lo que
  genero predicciones inestables en el conjunto de prueba y afecto la votacion mayoritaria.
- El mecanismo de `voting="hard"` da el mismo peso a todos los estimadores sin importar su calidad individual. Esto es
  una desventaja cuando los componentes tienen desempenos muy diferentes, como en este caso.
- Un ensamble mas efectivo para este problema seria un `StackingClassifier` donde GradientBoosting y RandomForest sean
  los estimadores base, o un `VotingClassifier` con `voting="soft"` usando modelos de calidad comparable.
- **Metodo seleccionado para implementacion:** GradientBoostingClassifier dentro de un Pipeline, por su mayor accuracy,
  F1-score y estabilidad frente al ensamble evaluado.

---

## Conclusiones generales

### Sobre el dataset

El dataset de Spotify representa un caso real de clasificacion multiclase con solapamiento natural entre categorias. La
principal dificultad no esta en la cantidad de datos ni en la calidad del preprocesamiento, sino en la naturaleza del
problema: los generos musicales son convenciones culturales, no categorias acusticamente puras. Esto establece un techo
de desempeno inherente que ningun modelo puede superar con estas variables sin incorporar informacion adicional como
letra, artista o contexto cultural. La correlacion entre `energy` y `acousticness` fue la relacion mas clara encontrada
entre variables independientes, mientras que el target `playlist_genre` no mostro correlacion lineal fuerte con ninguna
variable individual, lo que confirma que el problema requiere modelos capaces de capturar interacciones no lineales.

### Sobre la aplicacion de modelos en Python

El trabajo en Python con scikit-learn demostro que es posible construir un flujo completo de machine learning
reproducible y desplegable: desde la limpieza del dataset hasta la exposicion del modelo como API REST. GradientBoosting
fue el algoritmo que mejor capturo la complejidad del dataset gracias a su construccion secuencial de arboles, donde
cada iteracion corrige los errores de la anterior. El uso de Pipeline como unidad de serializacion garantizo que el
preprocesamiento y el modelo se apliquen de forma consistente en produccion, eliminando riesgos de inconsistencia entre
el entrenamiento y la inferencia. Un accuracy de ~0.59 en un problema de 6 clases con solapamiento acustico real se
considera un resultado razonable.

### Sobre el prototipado en Orange

Orange cumplio su funcion como herramienta exploratoria de bajo codigo: permitio comparar rapidamente multiples
algoritmos de forma visual sin necesidad de escribir codigo, identificar los modelos mas prometedores y obtener una
primera lectura de las metricas. Sin embargo, presenta limitaciones claras para proyectos con requisitos de despliegue:
no permite exportar el modelo entrenado de forma directa para integrarlo en una API, no soporta el uso de Pipelines de
preprocesamiento y su reproducibilidad es baja porque el flujo de trabajo es un archivo grafico, no codigo versionable.
Orange es util en etapas iniciales de exploracion, pero no reemplaza la implementacion en Python para proyectos que
requieren produccion.

### Sobre el ensamble

Los metodos de ensamble no garantizan mejoras automaticas sobre modelos individuales. Su efectividad depende de que los
estimadores base sean diversos y de calidad comparable. En este caso, incluir regresion logistica en un problema no
lineal y un arbol de decision sin restricciones genero un ensamble menos efectivo que el mejor modelo individual. La
leccion principal es que antes de aplicar un ensamble, cada componente debe evaluarse de forma independiente para
asegurar que aporta valor al conjunto.

### Sobre las dificultades del trabajo

- **Compatibilidad con PyCaret:** La incompatibilidad de PyCaret con Python 3.12 obligo a replantear la estrategia de
  comparacion de modelos. Se resolvio implementando la comparacion manualmente con scikit-learn, lo que en la practica
  ofrecio mayor control sobre el proceso.
- **Tiempo de entrenamiento de GradientBoosting:** Con `n_estimators=200`, el entrenamiento demoro considerablemente mas
  que RandomForest o Decision Tree. Para proyectos con mayor volumen de datos, seria necesario evaluar alternativas como
  XGBoost o LightGBM.
- **Convergencia de la regresion logistica:** Incluso con `max_iter=3000` y `solver="lbfgs"`, el modelo mostro
  advertencias de convergencia en algunos folds, lo que confirma que el problema no es linealmente separable.
- **Despliegue con Docker:** La configuracion del entorno de produccion requirio ajustar las versiones de dependencias
  para garantizar reproducibilidad entre el entorno local y el contenedor.
- **Solapamiento entre generos:** La mayor dificultad conceptual fue aceptar que un accuracy de ~0.59 no es un fallo del
  modelo sino una consecuencia de la naturaleza del problema. Generos como `pop` y `r&b` son categorias culturales con
  alta superposicion acustica real.

---

## Estado del proyecto

| Componente                    | Estado                               |
|-------------------------------|--------------------------------------|
| Data Cleaning                 | Completado                           |
| Analisis exploratorio         | Completado                           |
| Analisis de correlacion       | Completado                           |
| Prototipado en Orange         | Completado                           |
| Modelado con scikit-learn     | Completado                           |
| Matriz de confusion           | Completado                           |
| Metricas de efectividad       | Completado                           |
| Pipeline                      | Completado                           |
| Guardado del modelo (joblib)  | Completado                           |
| FastAPI (local)               | Completado                           |
| FastAPI (publico en servidor) | Completado - http://72.60.68.22:8013 |
| Dashboard Streamlit (publico) | Completado - http://72.60.68.22:8014 |
| Despliegue con Docker         | Completado                           |
| Ensamble (VotingClassifier)   | Completado                           |
| PyCaret                       | Incompatible con Python 3.12         |

---

## Pendientes / Limitaciones

- **PyCaret:** Requiere Python 3.10 o 3.11. No compatible con el entorno utilizado (Python 3.12).
- **Ajuste de hiperparametros sistematico:** No se aplico `GridSearchCV` ni `RandomizedSearchCV` sobre el modelo final.
  Queda como mejora futura.
- **Ensamble mejorado:** Un `StackingClassifier` o un `VotingClassifier` con estimadores de calidad comparable podria
  superar el resultado de GradientBoosting individual.
- **Balanceo de clases:** Se puede explorar SMOTE si el desbalance afecta el rendimiento en produccion.

---

## Referencias

- Ali, M. (2020). *PyCaret: An open source, low-code machine learning library in Python*. https://www.pycaret.org
- Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., Varoquaux, G. (2013). API design for
  machine learning software: experiences from the scikit-learn project. *ECML PKDD Workshop: Languages for Data Mining
  and Machine Learning*, 108-122.
- Demsar, J., Curk, T., Erjavec, A., Gorup, C., Hocevar, T., Milutinovic, M., Mozina, M., Polajnar, M., Toplak, M.,
  Staric, A., Stajdohar, M., Umek, L., Zagar, L., Zbontar, J., Zitnik, M., Zupan, B. (2013). Orange: Data mining toolbox
  in Python. *Journal of Machine Learning Research*, 14(1), 2349-2353.
- FastAPI. (s. f.). *Tutorial - User Guide*. https://fastapi.tiangolo.com/tutorial/
- PyCaret. (2024). *Installation*. https://pycaret.readthedocs.io/en/stable/installation.html
- PyCaret. (2024). *PyPI*. https://pypi.org/project/pycaret/
- Ramirez, S. (2018). *FastAPI*. https://fastapi.tiangolo.com
- scikit-learn. (s. f.). *Pipeline*. https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- scikit-learn. (s. f.).
  *VotingClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
- Docker. (s. f.). *Docker Compose*. https://docs.docker.com/compose/
