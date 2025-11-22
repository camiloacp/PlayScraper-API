from src.model.model import TopicModelingPipeline

def main():
    pipeline = TopicModelingPipeline(data_path="data/reviews_apps.csv", images_dir="images")
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()