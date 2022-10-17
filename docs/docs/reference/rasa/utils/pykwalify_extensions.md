---
sidebar_label: rasa.utils.pykwalify_extensions
title: rasa.utils.pykwalify_extensions
---
This module regroups custom validation functions, and it is
loaded as an extension of the pykwalify library:

https://pykwalify.readthedocs.io/en/latest/extensions.html#extensions

#### require\_response\_keys

```python
require_response_keys(responses: List[Dict[Text, Any]], rule_obj: Dict, path: Text) -> bool
```

Validate that response dicts have either the &quot;text&quot; key or the &quot;custom&quot; key.

