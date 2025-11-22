"""
Script de ejemplo para usar el modelo BERTopic entrenado y hacer predicciones.
"""

import logging
from src.model.model import TopicModelingPipeline
from src.settings.settings import IMAGES_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_from_saved_model(model_path="./model/topic_model.pkl"):
    """
    Ejemplo de cómo cargar un modelo guardado y hacer predicciones.
    
    Args:
        model_path: Ruta al modelo guardado
    """
    # Inicializar el pipeline (solo necesitamos el contenedor)
    pipeline = TopicModelingPipeline(data_path="", images_dir=IMAGES_DIR)
    
    # Cargar el modelo guardado
    pipeline.topic_model = TopicModelingPipeline.load_topic_model(model_path)
    
    # Ejemplos de nuevos documentos para predecir
    nuevos_documentos = [
        "Esta aplicación es excelente, me encanta la interfaz y es muy fácil de usar.",
        "La app se crashea todo el tiempo, muy frustrante.",
        "El servicio de delivery es rápido y eficiente.",
        "No puedo iniciar sesión, siempre me da error.",
        "Excelente para aprender cosas nuevas, los cursos son muy buenos.",
    ]
    
    # Hacer predicciones
    topics, probabilities = pipeline.predict_topics(nuevos_documentos)
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("PREDICCIONES DE TÓPICOS")
    print("="*80 + "\n")
    
    for i, (doc, topic, prob) in enumerate(zip(nuevos_documentos, topics, probabilities)):
        print(f"\n📄 Documento {i+1}:")
        print(f"   Texto: {doc}")
        print(f"   Tópico asignado: {topic}")
        
        if topic != -1:  # Si no es outlier
            # Obtener las palabras principales del tópico
            topic_info = pipeline.get_topic_info(topic)
            if topic_info is not None:
                top_words = " | ".join(topic_info.head(5)["Word"].tolist())
                print(f"   Palabras clave del tópico: {top_words}")
        else:
            print(f"   (Documento clasificado como outlier)")
        
        print(f"   Probabilidad: {prob:.4f}")
    
    # Mostrar información general de todos los tópicos
    print("\n" + "="*80)
    print("INFORMACIÓN DE TODOS LOS TÓPICOS")
    print("="*80 + "\n")
    
    all_topics_info = pipeline.get_topic_info()
    print(all_topics_info[["Topic", "Count", "Name"]].head(10))
    
    return topics, probabilities


def predict_single_document(document, model_path="./model/topic_model.pkl"):
    """
    Predicción para un solo documento.
    
    Args:
        document: String con el texto a clasificar
        model_path: Ruta al modelo guardado
        
    Returns:
        topic_id: ID del tópico asignado
        probability: Probabilidad de la asignación
    """
    # Inicializar el pipeline
    pipeline = TopicModelingPipeline(data_path="", images_dir=IMAGES_DIR)
    
    # Cargar el modelo
    pipeline.topic_model = TopicModelingPipeline.load_topic_model(model_path)
    
    # Hacer predicción
    topics, probabilities = pipeline.predict_topics(document)
    
    topic_id = topics[0]
    probability = probabilities[0]
    
    print(f"\n📄 Documento: {document}")
    print(f"   Tópico: {topic_id}")
    print(f"   Probabilidad: {probability:.4f}")
    
    if topic_id != -1:
        topic_info = pipeline.get_topic_info(topic_id)
        if topic_info is not None:
            print(f"   Palabras clave: {' | '.join(topic_info.head(5)['Word'].tolist())}")
    
    return topic_id, probability


if __name__ == "__main__":
    # Ejemplo 1: Predicción de múltiples documentos
    print("\n🔮 EJEMPLO 1: Predicción de múltiples documentos\n")
    predict_from_saved_model()
    
    # Ejemplo 2: Predicción de un solo documento
    print("\n\n🔮 EJEMPLO 2: Predicción de un solo documento\n")
    predict_single_document(
        "La aplicación de música tiene muy buena calidad de audio y muchas canciones"
    )
