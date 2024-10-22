from pymilvus import Collection
from pymilvus import connections


def query_milvus_collection() -> None:
    # Establish a connection to the Milvus server
    connections.connect(host="localhost", port="19530")

    # Assuming you have a collection instance
    collection = Collection("rasa_web")

    # Perform a query to retrieve specific fields
    res = collection.query(
        expr="source == 'https://rasa.com/docs/rasa-pro/'",
        output_fields=["text", "source"],
        limit=10,
        offset=0,
    )

    # Process the results
    for hit in res:
        print(hit)


if __name__ == "__main__":
    query_milvus_collection()
