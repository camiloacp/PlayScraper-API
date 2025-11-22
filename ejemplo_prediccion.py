"""
EJEMPLO RÁPIDO: Cómo usar el modelo BERTopic para hacer predicciones

Ejecuta este script después de haber entrenado el modelo con:
    python -m src.model.model
"""

from src.model.model import TopicModelingPipeline
from src.settings.settings import IMAGES_DIR


def main():
    print("\n" + "="*80)
    print("🔮 EJEMPLO DE PREDICCIÓN CON MODELO BERTOPIC")
    print("="*80 + "\n")
    
    # 1. Cargar el modelo guardado
    print("📂 Cargando modelo guardado...")
    pipeline = TopicModelingPipeline(data_path="", images_dir=IMAGES_DIR)
    pipeline.topic_model = TopicModelingPipeline.load_topic_model("./model/topic_model.pkl")
    
    # 2. Definir nuevos documentos para clasificar
    nuevos_reviews = [
        "La app es perfecta, me encanta la interfaz y es súper intuitiva",
        "Malísima aplicación, se cuelga constantemente y pierde mis datos",
        "El servicio de entrega es rápido, llegó antes de lo esperado",
        "No logro iniciar sesión, siempre me sale error de conexión",
        "Los cursos son excelentes, he aprendido muchísimo",
        "La calidad del audio es increíble, muy buena experiencia",
    ]
    
    # 3. Hacer predicciones
    print("\n🎯 Realizando predicciones...\n")
    topics, probabilities = pipeline.predict_topics(nuevos_reviews)
    
    # 4. Mostrar resultados
    print("="*80)
    print("RESULTADOS DE PREDICCIÓN")
    print("="*80 + "\n")
    
    for i, (review, topic, prob) in enumerate(zip(nuevos_reviews, topics, probabilities), 1):
        print(f"📝 Review {i}:")
        print(f"   Texto: \"{review}\"")
        print(f"   ├─ Tópico asignado: {topic}")
        print(f"   └─ Probabilidad: {prob:.2%}")
        
        # Si no es outlier, mostrar palabras clave del tópico
        if topic != -1:
            topic_info = pipeline.get_topic_info(topic_id=topic)
            if topic_info is not None and len(topic_info) > 0:
                top_5_words = ", ".join(topic_info.head(5)["Word"].tolist())
                print(f"      Palabras clave: {top_5_words}")
        else:
            print(f"      ⚠️  Clasificado como outlier (no pertenece claramente a ningún tópico)")
        print()
    
    # 5. Mostrar resumen de todos los tópicos disponibles
    print("\n" + "="*80)
    print("📊 RESUMEN DE TÓPICOS DISPONIBLES")
    print("="*80 + "\n")
    
    all_topics = pipeline.get_topic_info()
    print(all_topics[["Topic", "Count", "Name"]].to_string(index=False))
    
    print("\n✅ ¡Predicción completada!\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("\n❌ ERROR: No se encontró el modelo guardado.")
        print("   Por favor, entrena el modelo primero ejecutando:")
        print("   python -m src.model.model\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")

