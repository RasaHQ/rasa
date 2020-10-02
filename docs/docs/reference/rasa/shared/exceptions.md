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

## YamlSyntaxException Objects

```python
class YamlSyntaxException(RasaException)
```

Raised when a YAML file can not be parsed properly due to a syntax error.

