# Usage

## Requirements

Create a virtual environment and run `make install` to install Rasa Pro dependencies.
To run the `ingest_web.py` script, you must install first `pip install beautifulsoup4`.

## Ingesting Data
To ingest data from the web (e.g. Rasa online docs) run `python ingest_web.py -c config.yaml`.
Choose either `milvus_config.yaml` or `qdrant_config_web.yaml` to ingest data into Milvus or Qdrant respectively.

If you're using OpenAI embeddings, make sure that you have the OPENAI_API_KEY environment variables set.

Here are some pointers to make changes in the script,
- Ingest a different document format: change loader type in L38 of `ingest_web.py`.
- Change or add support for another embedding model: update the `embeddings_factory` function in `ingest_web.py`.

# Setting Up Qdrant

```
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

# Setting up Milvus

Run this command inside `milvus` directory which has the docker-compose.yml file [documentation](https://milvus.io/docs/install_standalone-docker.md)

```
docker compose up -d
```

## Create the Milvus Collection

Before ingesting data into Milvus, you need to create a collection. You can do this by running:

```
cd milvus
python milvus_collection.py
```

## Query the Milvus Collection

After ingesting data into Milvus, you can query the collection by running: `python milvus_query.py`.
