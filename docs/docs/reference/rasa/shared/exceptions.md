---
sidebar_label: rasa.shared.exceptions
title: rasa.shared.exceptions
---

## RasaException Objects

```python
class RasaException(Exception)
```

Base exception class for all errors raised by Rasa Open Source.

## RasaCoreException Objects

```python
class RasaCoreException(RasaException)
```

Basic exception for errors raised by Rasa Core.

## RasaXTermsError Objects

```python
class RasaXTermsError(RasaException)
```

Error in case the user didn&#x27;t accept the Rasa X terms.

## InvalidParameterException Objects

```python
class InvalidParameterException(RasaException,  ValueError)
```

Raised when an invalid parameter is used.

## YamlException Objects

```python
class YamlException(RasaException)
```

Raised if there is an error reading yaml.

#### \_\_init\_\_

```python
 | __init__(filename: Optional[Text] = None) -> None
```

Create exception.

**Arguments**:

- `filename` - optional file the error occurred in

## YamlSyntaxException Objects

```python
class YamlSyntaxException(YamlException)
```

Raised when a YAML file can not be parsed properly due to a syntax error.

## FileNotFoundException Objects

```python
class FileNotFoundException(RasaException,  FileNotFoundError)
```

Raised when a file, expected to exist, doesn&#x27;t exist.

## FileIOException Objects

```python
class FileIOException(RasaException)
```

Raised if there is an error while doing file IO.

## InvalidConfigException Objects

```python
class InvalidConfigException(ValueError,  RasaException)
```

Raised if an invalid configuration is encountered.

