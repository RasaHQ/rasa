---
sidebar_label: rasa.nlu.persistor
title: rasa.nlu.persistor
---
#### get\_persistor

```python
get_persistor(name: Text) -> Optional["Persistor"]
```

Returns an instance of the requested persistor.

Currently, `aws`, `gcs`, `azure` and providing module paths are supported remote
storages.

## Persistor Objects

```python
class Persistor(abc.ABC)
```

Store models in cloud and fetch them when needed.

#### persist

```python
 | persist(model_directory: Text, model_name: Text) -> None
```

Uploads a model persisted in the `target_dir` to cloud storage.

#### retrieve

```python
 | retrieve(model_name: Text, target_path: Text) -> None
```

Downloads a model that has been persisted to cloud storage.

## AWSPersistor Objects

```python
class AWSPersistor(Persistor)
```

Store models on S3.

Fetches them when needed, instead of storing them on the local disk.

## GCSPersistor Objects

```python
class GCSPersistor(Persistor)
```

Store models on Google Cloud Storage.

Fetches them when needed, instead of storing them on the local disk.

## AzurePersistor Objects

```python
class AzurePersistor(Persistor)
```

Store models on Azure

