import os
import pandas as pd
from datasets import Dataset
from src.settings.settings import EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from bertopic import BERTopic
from copy import deepcopy
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from dotenv import load_dotenv
import openai
from bertopic.representation import OpenAI

load_dotenv()

def load_data():
    df = pd.read_csv("data/reviews_apps.csv")
    return Dataset.from_pandas(df)

def train_model(data):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return embedding_model

def create_embeddings(embedding_model, data):
    embeddings = embedding_model.encode(data["content"], show_progress_bar=True)
    return embeddings

def reduce_dimensionality(embeddings, n_components=5):
    umap_model = UMAP(n_components=n_components, min_dist=0.0, metric="cosine", random_state=42)
    return umap_model, umap_model.fit_transform(embeddings)

def cluster_documents(reduced_embeddings):
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric="euclidean", cluster_selection_method="eom").fit(reduced_embeddings)
    return hdbscan_model, hdbscan_model.labels_

def create_df(reduced_embeddings, clusters):
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["cluster"] = [str(c) for c in clusters]
    clusters_df = df.loc[df.cluster != "-1", :]
    outliers_df = df.loc[df.cluster == "-1", :]
    return clusters_df, outliers_df

def plot_clusters(clusters_df, outliers_df):
    plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
    plt.scatter(clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int), alpha=0.6, s=2, cmap="tab20b")
    plt.axis("off")
    plt.savefig("images/clusters.png")

def train_topic_model(embedding_model, umap_model, hdbscan_model, data):
    topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True).fit(data["content"])
    return topic_model

def visualize_topic_model(topic_model):
    # Guardar visualización de topics (mapa de distancia entre topics)
    fig_topics = topic_model.visualize_topics(width=1200, height=800)
    fig_topics.write_image("images/topics.png")

    # Guardar gráfico de barras (palabras más importantes por topic)
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_image("images/barchart.png")

    # Guardar heatmap (similaridad entre topics)
    fig_heatmap = topic_model.visualize_heatmap(n_clusters=30)
    fig_heatmap.write_image("images/heatmap.png")

    # Guardar jerarquía (relaciones jerárquicas entre topics)
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_image("images/hierarchy.png")

def topic_differences(model, original_topics, nr_topics=5):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):

        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    return df.head(10)

def keybert_inspired_representation(topic_model):
    representation_model = KeyBERTInspired()
    topic_model.update_topics(data["content"], representation_model=representation_model)
    return topic_model

def maximal_marginal_relevance_representation(topic_model):
    representation_model = MaximalMarginalRelevance(diversity=0.5)
    topic_model.update_topics(data["content"], representation_model=representation_model)
    return topic_model

def openai_representation(topic_model):
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]

    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short topic label in the following format:
    topic: <short topic label>
    """

    # Update our topic representations using GPT-4o-mini
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    representation_model = OpenAI(
        client, model="gpt-4o-mini", exponential_backoff=True, chat=True, prompt=prompt
    )
    topic_model.update_topics(data["content"], representation_model=representation_model)
    return topic_model

def visualize_topic_model_with_documents(topic_model, data):
    fig = topic_model.visualize_document_datamap(
        data["content"],
        topics=list(range(25)),
        reduced_embeddings=reduced_embeddings,
        width=1200
    )
    fig.write_image("images/document_datamap.png")

if __name__ == "__main__":
    # Crear carpeta para imágenes si no existe
    os.makedirs("images", exist_ok=True)
    
    data = load_data()
    print(f"Data loaded successfully: {data}")
    embedding_model = train_model(data)
    print(f"Model trained successfully!")
    embeddings = create_embeddings(embedding_model, data)
    print(f"Embeddings created successfully: {embeddings.shape}")
    umap_model, reduced_embeddings = reduce_dimensionality(embeddings)
    print(f"Reduced embeddings successfully: {reduced_embeddings.shape}")
    hdbscan_model, clusters = cluster_documents(reduced_embeddings)
    print(f"Clusters created successfully!")
    print(f"Number of clusters: {len(set(clusters))}")
    _, reduced_embeddings_df = reduce_dimensionality(embeddings, n_components=2)
    clusters_df, outliers_df = create_df(reduced_embeddings_df, clusters)
    print(f"Clusters dataframe created successfully: {clusters_df.shape}")
    plot_clusters(clusters_df, outliers_df)
    print(f"Clusters plotted successfully!")

    topic_model = train_topic_model(embedding_model, umap_model, hdbscan_model, data)
    print(f"Topic model trained successfully!")
    
    visualize_topic_model(topic_model)
    print(f"All topic visualizations saved successfully!")

    original_topics = deepcopy(topic_model.get_topics())

    keybert_inspired_topic_model = keybert_inspired_representation(topic_model)
    print(f"KeyBERT inspired topic model trained successfully!")
    print(topic_differences(keybert_inspired_topic_model, original_topics))

    maximal_marginal_relevance_topic_model = maximal_marginal_relevance_representation(topic_model)
    print(f"Maximal marginal relevance topic model trained successfully!")
    print(topic_differences(maximal_marginal_relevance_topic_model, original_topics))

    openai_topic_model = openai_representation(topic_model)
    print(f"OpenAI topic model trained successfully!")
    print(topic_differences(openai_topic_model, original_topics))
