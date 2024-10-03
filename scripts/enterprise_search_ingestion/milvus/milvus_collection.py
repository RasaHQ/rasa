import logging

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType


logger = logging.getLogger(__name__)


def create_milvus_collection() -> None:
    """Creates a Milvus collection using a predefined schema."""
    # Connect to Milvus server
    connections.connect(host="localhost", port="19530")
    # Define fields
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Source URL of the webpage",
        ),
    ]
    # Enable dynamic schema in schema definition
    schema = CollectionSchema(
        fields, "Schema for online rasa docs", enable_dynamic_field=True
    )
    # Create the collection with dynamic schema enabled
    collection = Collection("rasa_web", schema)

    # Index the vector field and load the collection
    index_params = {"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}}

    collection.create_index(field_name="vector", index_params=index_params)
    # Load the collection
    collection.load()
    logger.info("Collection loaded successfully!")


if __name__ == "__main__":
    create_milvus_collection()
