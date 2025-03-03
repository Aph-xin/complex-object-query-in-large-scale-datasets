import argparse
import time
import pandas as pd
import torch
from system import VideoObjectQuerySystem
from utils import plot_and_save_dino_results, format_prompt


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Video Retrieval System with Milvus and OWL/DINO Predictor")
    
    # Logger
    parser.add_argument('--log-dir', type=str, default="../logs/cityscapes",
                        help="Directory to store log files")
    parser.add_argument('--log-filename', type=str, default="cityscapes_q12.log",
                        help="Log filename")

    # Database setup
    parser.add_argument('--milvus-host', type=str, default="localhost",
                        help="Milvus server host (default: localhost)")
    parser.add_argument('--milvus-port', type=str, default="19530",
                        help="Milvus server port (default: 19530)")
    parser.add_argument('--database-name', type=str, default="cityscapes_vit32_new",
                        help="Milvus database name (default: cityscapes_vit32_new)")

    # Dataset name
    parser.add_argument('--dataset', type=str, default="cityscapes",
                        help="Dataset name")
    parser.add_argument('--csv-path', type=str, default="../dataset/cityscapes/stuggart_overall.csv",
                        help="Path to the CSV file containing dataset info")

    # Model and query 
    parser.add_argument('--model-name', type=str, default="google/owlvit-base-patch32",
                        help="OWL Predictor model name (default: google/owlvit-base-patch32)")
    parser.add_argument('--image-encoder-engine', type=str, default=None,
                        help="Path to the image encoder engine file, and the example engine path: ../model/owl_image_encoder_patch32.engine")
    parser.add_argument('--prompts', type=str, default="a person carrying a black backpack, wearing blue jeans, white shirt, walking on the crosswalk",
                        help="Text prompt for querying")
    parser.add_argument('--model-prefix', type=str, default="Vit-B-32",
                        help="Model prefix for output naming")

    # Fast search and rerank
    parser.add_argument('--top-n', type=int, default=10,
                        help="Number of top similar images to retrieve")
    parser.add_argument('--top-k', type=int, default=10,
                        help="Number of top reranked results to save")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help="Threshold for OWL model (default: 0.1)")
    parser.add_argument('--interval', type=int, default=1,
                        help="Interval for frame uploading (default: 1)")

    # Rerank parameters
    parser.add_argument('--dino-config', type=str, default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to Grounding DINO config file")
    parser.add_argument('--dino-checkpoint', type=str, default="./GroundingDINO/weight/groundingdino_swint_ogc.pth",
                        help="Path to Grounding DINO checkpoint file")
    parser.add_argument('--dino-box-threshold', type=float, default=0.3,
                        help="Box threshold for DINO reranking (default: 0.3)")
    parser.add_argument('--dino-text-threshold', type=float, default=0.1,
                        help="Text threshold for DINO reranking (default: 0.1)")

    # Output directory
    parser.add_argument('--save-dir', type=str, default="../results/cityscapes_q12",
                        help="Directory to save results")

    # Whether to force rebuild the collection
    parser.add_argument('--force-rebuild', action='store_true',
                        help="Force rebuild the collection even if it exists (default: False)")

    return parser.parse_args()


def main():
    # Set CUDA device
    # torch.cuda.set_device(0)

    # Parse command-line arguments
    args = parse_args()

    # Initialize the system
    system = VideoObjectQuerySystem(
        log_dir=args.log_dir,
        log_filename=args.log_filename,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port
    )
    start_time = time.time()

    # Prepare data
    df = pd.read_csv(args.csv_path)

    # Initialize processor and search engine
    processor = system.setup_predictor(args.model_name, args.image_encoder_engine)
    search_engine = system.setup_search_engine(args.model_name, args.image_encoder_engine)

    # Check if collection needs to be created and uploaded
    collection = system.milvus_manager.load_collection(args.database_name, system.logger)
    if collection is None or args.force_rebuild:
        system.logger.info(f"Collection {args.database_name} does not exist or force rebuild is enabled. Creating and uploading data.")
        collection = system.milvus_manager.create_collection(args.database_name, 512, system.logger)
        system.upload_frames(collection, processor, df, args.interval)
    else:
        system.logger.info(f"Collection {args.database_name} already exists. Loading existing collection.")

    # Format prompts
    owl_prompt = format_prompt(args.prompts, "owl")
    dino_prompt = format_prompt(args.prompts, "dino")

    # Log query parameters
    system.logger.info(f"Using OWL prompt: {owl_prompt}")
    system.logger.info(f"Using DINO prompt: {dino_prompt}")
    system.logger.info(f"Using rerank model: {args.model_prefix}")
    system.logger.info(f"Using output path: {args.save_dir}")

    # Execute query
    text_output = processor.encode_query(owl_prompt)
    results = search_engine.search_similar_images(text_output, collection, args.top_n)

    # Perform rerank with DINO
    final_result = search_engine.unique_dino_reranker(
        config_file=args.dino_config,
        checkpoint_path=args.dino_checkpoint,
        results=results,
        prompts=dino_prompt,  # Use DINO format
        box_threshold=args.dino_box_threshold,
        text_threshold=args.dino_text_threshold,
        top_k=args.top_k,
        save_dir=args.save_dir
    )

    # Save DINO reranked result images
    plot_and_save_dino_results(final_result, args.save_dir)

    # Log total execution time
    total_time = time.time() - start_time
    system.logger.info(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()