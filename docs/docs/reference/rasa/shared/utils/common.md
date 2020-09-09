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

#### lazy\_property

```python
lazy_property(function: Callable) -> Any
```

Allows to avoid recomputing a property over and over.

The result gets stored in a local var. Computation of the property
will happen once, on the first call of the property. All
succeeding calls will use the value stored in the private property.

