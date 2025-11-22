from google_play_scraper import app, Sort, reviews, reviews_all
import pandas as pd
import regex as re
from settings.settings import URLS_APPS
from utils.utils import extract_app_id

app_id = [extract_app_id(url) for url in URLS_APPS]

def extract_data(app_id):
    info = app(app_id)
    all_reviews, continuation_token = reviews(
        app_id,
        lang="es",
        sort=Sort.MOST_RELEVANT,
        country="co",
        count=10_000
    )
    print(f"Total de comentarios de la app {info['title']}: {len(all_reviews)}")
    return all_reviews

all_data = []
for id in app_id:
    all_data.extend(extract_data(id))
df_final = pd.DataFrame(all_data)
df_final = df_final[['score', 'content']]

print(df_final.shape)
print(df_final.head())

df_final.to_csv("data/reviews_apps.csv", index=False)