import os
URLS_APPS = [
    "https://play.google.com/store/apps/details?id=com.roblox.client", # Roblox
    "https://play.google.com/store/apps/details?id=com.openai.chatgpt&hl=en", # ChatGPT
    "https://play.google.com/store/apps/details?id=com.mercadolibre",  # MercadoLibre
    "https://play.google.com/store/apps/details?id=com.nequi.MobileApp&hl=en", # Nequi
    "https://play.google.com/store/apps/details?id=com.nu.production",  # Nu Bank
    "https://play.google.com/store/apps/details?id=com.whatsapp",  # WhatsApp
    "https://play.google.com/store/apps/details?id=com.platzi.platzi&hl=en",  # Platzi
    "https://play.google.com/store/apps/details?id=com.udemy.android",  # Udemy
    "https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically&hl=en",  # TikTok
    "https://play.google.com/store/apps/details?id=com.zzkko&hl=en",  # SHEIN
    "https://play.google.com/store/apps/details?id=com.spotify.music&hl=en", # Spotify
    "https://play.google.com/store/apps/details?id=com.netflix.mediaclient",  # Netflix
    "https://play.google.com/store/apps/details?id=com.ubercab",  # Uber
    "https://play.google.com/store/apps/details?id=com.grability.rappi",  # Rappi
]

NUM_REVIEWS = 10_0000
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# --- Constants and Configuration ---
DATA_PATH = "data/reviews_apps.csv"
IMAGES_DIR = "images"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UMAP_COMPONENTS_VIZ = 2
UMAP_COMPONENTS_CLUSTER = 5
HDBSCAN_MIN_CLUSTER_SIZE = 5
RANDOM_STATE = 42