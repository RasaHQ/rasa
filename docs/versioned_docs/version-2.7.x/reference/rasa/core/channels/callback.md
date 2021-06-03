---
sidebar_label: rasa.core.channels.callback
title: rasa.core.channels.callback
---
## CallbackInput Objects

```python
class CallbackInput(RestInput)
```

A custom REST http input channel that responds using a callback server.

Incoming messages are received through a REST interface. Responses
are sent asynchronously by calling a configured external REST endpoint.

