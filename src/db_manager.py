from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from typing import List, Optional
from logger import Logger


class MilvusManager:
    """Milvus database management class"""
    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        """Connect to Milvus"""
        connections.connect("default", host=self.host, port=self.port)

    def create_collection(self, collection_name: str, dim: int, logger: Logger) -> Collection:
        """Create a Milvus collection"""
        if utility.has_collection(collection_name):
            logger.info(f"Dropping existing collection {collection_name}")
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="image_embeds", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="pred_boxes", dtype=DataType.FLOAT_VECTOR, dim=4)
        ]
        schema = CollectionSchema(fields=fields, description="Image embeddings with predicted boxes")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 512}}
        collection.create_index("image_embeds", index_params)
        collection.create_index("pred_boxes", index_params)
        collection.load()
        logger.info(f"Collection {collection_name} created successfully with embedding dimension {dim}")
        return collection

    def load_collection(self, collection_name: str, logger: Logger) -> Optional[Collection]:
        """Load an existing Milvus collection"""
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists. Loading the collection.")
            collection = Collection(collection_name)
            collection.load()
            logger.info(f"Collection {collection_name} loaded successfully.")
            return collection
        return None

    def insert_data(self, collection: Collection, data: List[dict], logger: Logger):
        """Insert data into Milvus"""
        try:
            collection.insert(data)
        except Exception as e:
            logger.error(f"Error inserting to Milvus: {e}")

    def check_image_id_exists(self, collection: Collection, image_id: str) -> bool:
        """Check if an image ID exists"""
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        expr = f'image_id == "{image_id}"'
        results = collection.search(
            [[0.0] * 512], anns_field="image_embeds", param=search_params, limit=1, expr=expr
        )
        return len(results[0]) > 0