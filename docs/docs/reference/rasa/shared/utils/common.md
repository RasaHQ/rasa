---
sidebar_label: rasa.shared.utils.common
title: rasa.shared.utils.common
---

#### class\_from\_module\_path

```python
class_from_module_path(module_path: Text, lookup_path: Optional[Text] = None) -> Any
```

Given the module name and path of a class, tries to retrieve the class.

The loaded class can be used to instantiate new objects.

#### all\_subclasses

```python
all_subclasses(cls: Any) -> List[Any]
```

Returns all known (imported) subclasses of a class.

#### module\_path\_from\_instance

```python
module_path_from_instance(inst: Any) -> Text
```

Return the module path of an instance&#x27;s class.

