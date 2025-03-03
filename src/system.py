import os
import re
import pandas as pd
from tqdm import tqdm
from encoder.owl_reranker import OwlPredictor
from logger import Logger
from pymilvus import Collection
from db_manager import MilvusManager
from frame_processor import FrameProcessor
from search_engine import SearchEngine


class VideoObjectQuerySystem:
    """Video retrieval system main class"""
    def __init__(self, log_dir: str, log_filename: str, milvus_host: str = "localhost", milvus_port: str = "19530"):
        self.logger = Logger(log_dir, log_filename)
        self.milvus_manager = MilvusManager(milvus_host, milvus_port)

    def setup_predictor(self, model_name: str, image_encoder_engine: str) -> FrameProcessor:
        """Initialize frame processor"""
        predictor = OwlPredictor(model_name=model_name, image_encoder_engine=image_encoder_engine)
        return FrameProcessor(predictor)

    def setup_search_engine(self, model_name: str, image_encoder_engine: str) -> SearchEngine:
        """Initialize search engine"""
        predictor = OwlPredictor(model_name=model_name, image_encoder_engine=image_encoder_engine)
        return SearchEngine(predictor, self.logger)

    def process_adjacent_frames(self, image_path: str, collection: Collection, processor: FrameProcessor, num_frames: int = 10):
        """Process adjacent frames"""
        if self.milvus_manager.check_image_id_exists(collection, image_path):
            self.logger.info(f"Frame {image_path} already exists in the database. Skipping.")
            return

        processor.encode_and_store(image_path, collection, self.logger)
        match = re.search(r'_(\d{6})_leftImg8bit\.png$', image_path)
        if not match:
            self.logger.error(f"Failed to extract frame number from {image_path}")
            return

        frame_number = int(match.group(1))
        adjacent_frame_numbers = [n for n in range(frame_number - num_frames, frame_number + num_frames + 1) if n >= 0]
        base_dir, base_name = os.path.dirname(image_path), re.sub(r'_(\d{6})_leftImg8bit\.png$', '', os.path.basename(image_path))
        processed_frames = set()

        for number in adjacent_frame_numbers:
            frame = os.path.join(base_dir, f"{base_name}_{str(number).zfill(6)}_leftImg8bit.png")
            if frame in processed_frames:
                continue
            processed_frames.add(frame)
            if os.path.exists(frame):
                self.logger.info(f"Processing frame {frame}")
                if not self.milvus_manager.check_image_id_exists(collection, frame):
                    processor.encode_and_store(frame, collection, self.logger)
                else:
                    self.logger.info(f"Frame {frame} already exists in Milvus. Skipping upload.")
            else:
                self.logger.warning(f"Frame {frame} does not exist.")

    def upload_frames(self, collection: Collection, processor: FrameProcessor, df: pd.DataFrame, interval: int):
        """Upload frame data"""
        for index in tqdm(range(0, df.shape[0], interval), total=(df.shape[0] // interval), desc="Inserting frames"):
            image_path = df.iloc[index]['path']
            processor.encode_and_store(image_path, collection, self.logger)
        self.logger.info(f"All the {df.shape[0]} frames uploaded in {collection.name} successfully")