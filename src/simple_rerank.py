import time
import pandas as pd
from system import VideoObjectQuerySystem


def main():
    # Initialize the system
    system = VideoObjectQuerySystem(log_dir="../logs/cityscapes", log_filename="cityscapes_q11.log")
    start_time = time.time()

    # Prepare the data
    database_name = "cityscapes_vit32_new"
    dataset = "cityscapes"
    df = pd.read_csv('/home/aph/nanoowl/dataset/cityscapes/stuggart_overall.csv')
    top_n, top_k, threshold, interval = 2, 2, 0.05, 1

    # Initialize the processor and search engine
    processor = system.setup_predictor("google/owlvit-base-patch32", "../model/owl_image_encoder_patch32.engine")
    search_engine = system.setup_search_engine("google/owlvit-base-patch32", "../model/owl_image_encoder_patch32.engine")

    # Create or load the collection
    collection = system.milvus_manager.create_collection(database_name, 512, system.logger)
    system.upload_frames(collection, processor, df, interval)
    collection = system.milvus_manager.load_collection(database_name, system.logger)

    # Set query parameters
    prompts = "[a person walking on the street]"
    model_prefix = "Vit-B-32"
    save_dir = "../results/cityscapes_q11"

    system.logger.info(f"Using prompt: {prompts}")
    system.logger.info(f"Using rerank model: {model_prefix}")
    system.logger.info(f"Using output path: {save_dir}")

    # Execute query and re-ranking
    text_output = processor.encode_query(prompts)
    results = search_engine.search_similar_images(text_output, collection, top_n)
    text_output_rerank = processor.encode_query(prompts)
    search_engine.unique_reranker(results, prompts, text_output_rerank, threshold, top_k, save_dir)

    # Record total execution time
    total_time = time.time() - start_time
    system.logger.info(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
