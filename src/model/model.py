import os
import pandas as pd
from datasets import Dataset
from src.settings.settings import EMBEDDING_MODEL, UMAP_COMPONENTS_CLUSTER, UMAP_COMPONENTS_VIZ, HDBSCAN_MIN_CLUSTER_SIZE, RANDOM_STATE, DATA_PATH, IMAGES_DIR, OPENAI_API_KEY
import logging
import matplotlib.pyplot as plt
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI as BERTopicOpenAI
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class TopicModelingPipeline:
    """
    A class to encapsulate the topic modeling pipeline, from data loading to visualization.
    """
    def __init__(self, data_path, images_dir):
        self.data_path = data_path
        self.images_dir = images_dir
        self.data = None
        self.embedding_model = None
        self.embeddings = None
        self.umap_model = None
        self.reduced_embeddings = None
        self.reduced_embeddings_2d = None
        self.hdbscan_model = None
        self.clusters = None
        self.topic_model = None
        
        os.makedirs(self.images_dir, exist_ok=True)
        if not OPENAI_API_KEY:
            logging.warning("OPENAI_API_KEY environment variable not set. OpenAI representation will not work.")
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def load_data(self):
        """Loads data from a CSV file."""
        try:
            logging.info(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path)
            self.data = Dataset.from_pandas(df)
            logging.info("Data loaded successfully.")
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {self.data_path}")
            raise

    def train_embedding_model(self):
        """Initializes and trains the sentence transformer model."""
        logging.info("Training embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logging.info("Embedding model trained.")

    def create_embeddings(self):
        """Creates embeddings for the dataset content."""
        if self.data is None or self.embedding_model is None:
            raise ValueError("Data and embedding model must be loaded first.")
        logging.info("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(self.data["content"], show_progress_bar=True)
        logging.info(f"Embeddings created with shape: {self.embeddings.shape}")

    def reduce_dimensionality(self, n_components):
        """Reduces dimensionality of embeddings using UMAP."""
        logging.info(f"Reducing dimensionality to {n_components} components...")
        umap_model = UMAP(n_components=n_components, min_dist=0.0, metric="cosine", random_state=RANDOM_STATE)
        reduced_embeddings = umap_model.fit_transform(self.embeddings)
        logging.info(f"Dimensionality reduced. New shape: {reduced_embeddings.shape}")
        return umap_model, reduced_embeddings

    def cluster_documents(self, reduced_embeddings):
        """Clusters documents using HDBSCAN."""
        logging.info("Clustering documents...")
        hdbscan_model = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, metric="euclidean", cluster_selection_method="eom").fit(reduced_embeddings)
        self.hdbscan_model = hdbscan_model
        self.clusters = hdbscan_model.labels_
        logging.info(f"Clustering complete. Found {len(set(self.clusters))} clusters.")

    def plot_clusters(self, reduced_embeddings_for_plot):
        """Plots and saves the document clusters."""
        logging.info("Plotting clusters...")
        df = pd.DataFrame(reduced_embeddings_for_plot, columns=["x", "y"])
        df["cluster"] = [str(c) for c in self.clusters]
        clusters_df = df.loc[df.cluster != "-1", :]
        outliers_df = df.loc[df.cluster == "-1", :]

        plt.figure(figsize=(10, 8))
        plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
        plt.scatter(clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int), alpha=0.6, s=2, cmap="tab20b")
        plt.axis("off")
        
        plot_path = os.path.join(self.images_dir, "clusters.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Cluster plot saved to {plot_path}")

    def train_topic_model(self):
        """Trains the BERTopic model."""
        logging.info("Training BERTopic model...")
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            verbose=True
        ).fit(self.data["content"])
        
        # Log topic information
        num_topics = len(set(self.topic_model.topics_)) - (1 if -1 in self.topic_model.topics_ else 0)
        logging.info(f"BERTopic model trained. Found {num_topics} topics (excluding outliers).")
        
        if num_topics < 3:
            logging.warning(f"Only {num_topics} topics found. Some visualizations may fail due to insufficient topics.")

    def visualize_topic_model(self, topic_model=None, suffix=""):
        """Generates and saves all standard BERTopic visualizations."""
        if topic_model is None:
            topic_model = self.topic_model # Usar el modelo principal si no se pasa uno

        if topic_model is None:
            logging.warning("Topic model not available. Skipping visualization.")
            return

        num_topics = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
        logging.info(f"Generating visualizations for model with {num_topics} topics (suffix: '{suffix}')...")

        visualizations_to_create = []
        
        # visualize_topics requires at least 3 topics for UMAP dimensionality reduction
        if num_topics >= 3:
            visualizations_to_create.append(("topics", lambda: topic_model.visualize_topics(width=1200, height=800)))
        else:
            logging.warning(f"Skipping 'topics' visualization: requires at least 3 topics, found {num_topics}")
        
        # Other visualizations
        if num_topics > 0:
            visualizations_to_create.append(("barchart", lambda: topic_model.visualize_barchart(top_n_topics=min(10, num_topics))))
            visualizations_to_create.append(("heatmap", lambda: topic_model.visualize_heatmap(n_clusters=min(30, num_topics))))
            visualizations_to_create.append(("hierarchy", lambda: topic_model.visualize_hierarchy()))

        # Generate and save each visualization
        for name, viz_func in visualizations_to_create:
            try:
                logging.info(f"Creating {name} visualization...")
                fig = viz_func()
                # Añadir sufijo al nombre del archivo
                path = os.path.join(self.images_dir, f"{name}{suffix}.png")
                fig.write_image(path)
                logging.info(f"Saved {name} visualization to {path}")
            except Exception as e:
                logging.error(f"Failed to create/save {name} visualization: {e}")
    
    def visualize_document_datamap(self, topic_model=None, reduced_embeddings_2d=None, suffix="", top_n_topics=20, width=1200, save_path=None):
        """
        Generates and saves a document datamap visualization.
        
        Args:
            topic_model: The topic model to use for visualization.
            reduced_embeddings_2d: 2D reduced embeddings for visualization (required).
            suffix: A suffix to add to the output filename.
            top_n_topics: Number of top topics to visualize (default: 20)
            width: Width of the visualization in pixels
            save_path: Optional custom path to save the image. If None, saves to images_dir
            
        Returns:
            The matplotlib figure object
        """
        if topic_model is None:
            topic_model = self.topic_model

        if topic_model is None:
            logging.warning("Topic model not trained. Skipping document datamap visualization.")
            return None
        
        if reduced_embeddings_2d is None:
            logging.warning("2D reduced embeddings not provided. Skipping document datamap visualization.")
            return None
        
        try:
            logging.info("Generating document datamap visualization...")
            
            # Create titles from the first 50 characters of each document
            titles = [doc[:50] + "..." if len(doc) > 50 else doc for doc in self.data["content"]]
            
            # Get the number of available topics
            num_topics = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
            topics_to_show = min(top_n_topics, num_topics)
            
            # Generate the visualization
            fig = topic_model.visualize_document_datamap(
                titles,
                topics=list(range(topics_to_show)),
                reduced_embeddings=reduced_embeddings_2d,
                width=width
            )
            
            # Save the figure
            if save_path is None:
                save_path = os.path.join(self.images_dir, f"document_datamap{suffix}.png")
            
            # Save using matplotlib (datamapplot returns a matplotlib-compatible figure)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Document datamap saved to {save_path}")
            return fig
            
        except Exception as e:
            logging.error(f"Failed to create document datamap visualization: {e}")
            return None

    def update_and_show_representation(self, representation_model, name):
        """Updates topic representation with a given model and prints differences."""
        # Check if we have enough topics
        num_topics = len(set(self.topic_model.topics_)) - (1 if -1 in self.topic_model.topics_ else 0)
        
        if num_topics < 2:
            logging.warning(f"Skipping {name} representation update: requires at least 2 topics, found {num_topics}")
            return None
        
        logging.info(f"Updating topic representation using {name}...")
        
        try:
            original_topics = deepcopy(self.topic_model.get_topics())
            
            # Create a new topic model instance for the updated representation
            updated_topic_model = deepcopy(self.topic_model)
            updated_topic_model.update_topics(self.data["content"], representation_model=representation_model)

            logging.info(f"Differences for {name}:")
            print(self.topic_differences(updated_topic_model, original_topics))
            return updated_topic_model
        except Exception as e:
            logging.error(f"Failed to update topic representation with {name}: {e}")
            return None

    def topic_differences(self, model, original_topics, nr_topics=5):
        """Shows the differences in topic representations between two models."""
        df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
        for topic in range(nr_topics):
            if topic in original_topics and model.get_topic(topic):
                og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
                new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
                df.loc[len(df)] = [topic, og_words, new_words]
        return df.head(10)
    
    def save_topic_model(self, path="./model/topic_model.pkl"):
        """Saves the topic model to a pickle file."""
        logging.info(f"Saving topic model to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.topic_model.save(path, serialization="pickle", save_ctfidf=True, save_embedding_model=self.embedding_model)
        logging.info("Topic model saved successfully.")
    
    @staticmethod
    def load_topic_model(path="./model/topic_model.pkl"):
        """Loads a saved topic model from a pickle file."""
        logging.info(f"Loading topic model from {path}...")
        topic_model = BERTopic.load(path)
        logging.info("Topic model loaded successfully.")
        return topic_model
    
    def predict_topics(self, documents):
        """
        Predicts topics for new documents.
        
        Args:
            documents: List of strings or a single string
            
        Returns:
            topics: List of topic IDs for each document
            probabilities: List of probability distributions for each document
        """
        if self.topic_model is None:
            raise ValueError("Topic model not trained or loaded. Train a model or load one first.")
        
        # Ensure documents is a list
        if isinstance(documents, str):
            documents = [documents]
        
        logging.info(f"Predicting topics for {len(documents)} documents...")
        topics, probabilities = self.topic_model.transform(documents)
        
        return topics, probabilities
    
    def get_topic_info(self, topic_id=None):
        """
        Gets information about topics.
        
        Args:
            topic_id: Optional. If provided, returns info for that specific topic.
                     If None, returns info for all topics.
        
        Returns:
            DataFrame with topic information
        """
        if self.topic_model is None:
            raise ValueError("Topic model not trained or loaded.")
        
        if topic_id is not None:
            # Get specific topic words
            topic_words = self.topic_model.get_topic(topic_id)
            if topic_words:
                return pd.DataFrame(topic_words, columns=["Word", "Score"])
            else:
                logging.warning(f"Topic {topic_id} not found.")
                return None
        else:
            # Get all topics info
            return self.topic_model.get_topic_info()

    def run_pipeline(self):
        """Executes the full topic modeling pipeline."""
        self.load_data()
        self.train_embedding_model()
        self.create_embeddings()
        
        self.umap_model, self.reduced_embeddings = self.reduce_dimensionality(n_components=UMAP_COMPONENTS_CLUSTER)
        _, self.reduced_embeddings_2d = self.reduce_dimensionality(n_components=UMAP_COMPONENTS_VIZ)
        
        self.cluster_documents(self.reduced_embeddings)
        self.plot_clusters(self.reduced_embeddings_2d)
        
        self.train_topic_model()

        # --- 1. Visualización del modelo ORIGINAL ---
        logging.info("--- Visualizing Original Model ---")
        self.visualize_topic_model(suffix="_original")
        self.visualize_document_datamap(reduced_embeddings_2d=self.reduced_embeddings_2d, suffix="_original")

        # --- 2. Actualización y Visualización con KeyBERT ---
        logging.info("--- Updating and Visualizing with KeyBERTInspired ---")
        keybert_model = self.update_and_show_representation(KeyBERTInspired(), "KeyBERTInspired")
        if keybert_model:
            self.visualize_topic_model(topic_model=keybert_model, suffix="_keybert")
            self.visualize_document_datamap(topic_model=keybert_model, reduced_embeddings_2d=self.reduced_embeddings_2d, suffix="_keybert")

        # --- 3. Actualización y Visualización con MMR ---
        logging.info("--- Updating and Visualizing with MaximalMarginalRelevance ---")
        mmr_model = self.update_and_show_representation(MaximalMarginalRelevance(diversity=0.5), "MaximalMarginalRelevance")
        if mmr_model:
            self.visualize_topic_model(topic_model=mmr_model, suffix="_mmr")
            self.visualize_document_datamap(topic_model=mmr_model, reduced_embeddings_2d=self.reduced_embeddings_2d, suffix="_mmr")

        # --- 4. Actualización y Visualización con OpenAI ---
        if self.openai_client.api_key:
            logging.info("--- Updating and Visualizing with OpenAI ---")
            prompt = """
            I have a topic that contains the following documents: [DOCUMENTS]
            The topic is described by the following keywords: [KEYWORDS]
            Based on the information above, extract a short topic label in the following format:
            topic: <short topic label>
            """
            openai_repr_model = BERTopicOpenAI(
                self.openai_client, model="gpt-4o-mini", exponential_backoff=True, chat=True, prompt=prompt
            )
            openai_model = self.update_and_show_representation(openai_repr_model, "OpenAI")
            if openai_model:
                self.visualize_topic_model(topic_model=openai_model, suffix="_openai")
                self.visualize_document_datamap(topic_model=openai_model, reduced_embeddings_2d=self.reduced_embeddings_2d, suffix="_openai")

        self.save_topic_model()

if __name__ == "__main__":
    pipeline = TopicModelingPipeline(data_path=DATA_PATH, images_dir=IMAGES_DIR)
    pipeline.run_pipeline()