# Topic Modeling - Guía de Uso

Este directorio contiene el pipeline de modelado de tópicos usando BERTopic.

## 📁 Archivos

- `model.py`: Clase principal `TopicModelingPipeline` con todo el pipeline
- `predict.py`: Scripts de ejemplo para hacer predicciones con modelos guardados

## 🚀 Uso

### 1. Entrenar un modelo nuevo

```python
from src.model.model import TopicModelingPipeline
from src.settings.settings import DATA_PATH, IMAGES_DIR

# Crear y ejecutar el pipeline
pipeline = TopicModelingPipeline(data_path=DATA_PATH, images_dir=IMAGES_DIR)
pipeline.run_pipeline()

# El modelo se guardará automáticamente en ./model/topic_model.pkl
```

O desde la terminal:

```bash
python -m src.model.model
```

### 2. Cargar un modelo guardado

```python
from src.model.model import TopicModelingPipeline

# Opción 1: Usar el método estático
topic_model = TopicModelingPipeline.load_topic_model("./model/topic_model.pkl")

# Opción 2: Cargar en un pipeline existente
pipeline = TopicModelingPipeline(data_path="", images_dir="images")
pipeline.topic_model = TopicModelingPipeline.load_topic_model("./model/topic_model.pkl")
```

### 3. Hacer predicciones con nuevos documentos

#### Múltiples documentos:

```python
from src.model.model import TopicModelingPipeline

# Inicializar y cargar modelo
pipeline = TopicModelingPipeline(data_path="", images_dir="images")
pipeline.topic_model = TopicModelingPipeline.load_topic_model("./model/topic_model.pkl")

# Nuevos documentos
documentos = [
    "Esta aplicación es excelente",
    "La app se crashea mucho",
    "El delivery es muy rápido"
]

# Predecir tópicos
topics, probabilities = pipeline.predict_topics(documentos)

# Resultados
for doc, topic, prob in zip(documentos, topics, probabilities):
    print(f"Documento: {doc}")
    print(f"Tópico: {topic}, Probabilidad: {prob:.4f}\n")
```

#### Un solo documento:

```python
from src.model.model import TopicModelingPipeline

pipeline = TopicModelingPipeline(data_path="", images_dir="images")
pipeline.topic_model = TopicModelingPipeline.load_topic_model("./model/topic_model.pkl")

documento = "La calidad de las entregas es excelente"
topics, probs = pipeline.predict_topics(documento)

print(f"Tópico asignado: {topics[0]}")
print(f"Probabilidad: {probs[0]:.4f}")
```

### 4. Obtener información de los tópicos

#### Ver todos los tópicos:

```python
# Obtener información de todos los tópicos
topic_info = pipeline.get_topic_info()
print(topic_info.head(10))
```

#### Ver un tópico específico:

```python
# Obtener palabras clave de un tópico específico
topic_id = 0
topic_words = pipeline.get_topic_info(topic_id=topic_id)
print(f"\nPalabras del tópico {topic_id}:")
print(topic_words)
```

### 5. Usar el script de predicción de ejemplo

```bash
# Ejecutar ejemplos de predicción
python -m src.model.predict
```

## 🔧 Métodos principales

### `TopicModelingPipeline`

#### Entrenamiento:

- `load_data()`: Carga datos desde CSV
- `train_embedding_model()`: Inicializa el modelo de embeddings
- `create_embeddings()`: Crea embeddings para los documentos
- `reduce_dimensionality(n_components)`: Reduce dimensionalidad con UMAP
- `cluster_documents(reduced_embeddings)`: Agrupa documentos con HDBSCAN
- `train_topic_model()`: Entrena el modelo BERTopic
- `visualize_topic_model()`: Genera visualizaciones
- `run_pipeline()`: Ejecuta todo el pipeline completo

#### Inferencia:

- `save_topic_model(path)`: Guarda el modelo entrenado
- `load_topic_model(path)`: [Estático] Carga un modelo guardado
- `predict_topics(documents)`: Predice tópicos para nuevos documentos
- `get_topic_info(topic_id)`: Obtiene información de tópicos

## 📊 Visualizaciones generadas

El pipeline genera automáticamente las siguientes visualizaciones en `images/`:

- `topics.png`: Visualización 2D de tópicos (requiere ≥3 tópicos)
- `barchart.png`: Gráfico de barras con los tópicos más frecuentes
- `heatmap.png`: Mapa de calor de similitud entre tópicos
- `hierarchy.png`: Jerarquía de tópicos
- `clusters.png`: Visualización de clusters de documentos

## ⚙️ Configuración

Los parámetros del modelo se configuran en `src/settings/settings.py`:

```python
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
UMAP_COMPONENTS_CLUSTER = 5
UMAP_COMPONENTS_VIZ = 2
HDBSCAN_MIN_CLUSTER_SIZE = 5  # Ajustar según tamaño del dataset
RANDOM_STATE = 42
```

### Recomendaciones para `HDBSCAN_MIN_CLUSTER_SIZE`:

- Dataset pequeño (< 1,000 documentos): 5-15
- Dataset mediano (1,000-10,000): 15-50
- Dataset grande (> 10,000): 50-100

## 🎯 Ejemplos de uso completo

Ver `predict.py` para ejemplos completos de:

1. Predicción de múltiples documentos
2. Predicción de un solo documento
3. Análisis de tópicos asignados
