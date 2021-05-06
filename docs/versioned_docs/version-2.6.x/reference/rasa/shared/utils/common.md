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

**Arguments**:

- `module_path` - either an absolute path to a Python class,
  or the name of the class in the local / global scope.
- `lookup_path` - a path where to load the class from, if it cannot
  be found in the local / global scope.
  

**Returns**:

  a Python class
  

**Raises**:

  ImportError, in case the Python class cannot be found.

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

#### sort\_list\_of\_dicts\_by\_first\_key

```python
sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]
```

Sorts a list of dictionaries by their first key.

#### lazy\_property

```python
lazy_property(function: Callable) -> Any
```

Allows to avoid recomputing a property over and over.

The result gets stored in a local var. Computation of the property
will happen once, on the first call of the property. All
succeeding calls will use the value stored in the private property.

#### cached\_method

```python
cached_method(f: Callable[..., Any]) -> Callable[..., Any]
```

Caches method calls based on the call&#x27;s `args` and `kwargs`.

Works for `async` and `sync` methods. Don&#x27;t apply this to functions.

**Arguments**:

- `f` - The decorated method whose return value should be cached.
  

**Returns**:

  The return value which the method gives for the first call with the given
  arguments.

#### transform\_collection\_to\_sentence

```python
transform_collection_to_sentence(collection: Collection[Text]) -> Text
```

Transforms e.g. a list like [&#x27;A&#x27;, &#x27;B&#x27;, &#x27;C&#x27;] into a sentence &#x27;A, B and C&#x27;.

#### minimal\_kwargs

```python
minimal_kwargs(kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None) -> Dict[Text, Any]
```

Returns only the kwargs which are required by a function. Keys, contained in
the exception list, are not included.

**Arguments**:

- `kwargs` - All available kwargs.
- `func` - The function which should be called.
- `excluded_keys` - Keys to exclude from the result.
  

**Returns**:

  Subset of kwargs which are accepted by `func`.

#### mark\_as\_experimental\_feature

```python
mark_as_experimental_feature(feature_name: Text) -> None
```

Warns users that they are using an experimental feature.

#### arguments\_of

```python
arguments_of(func: Callable) -> List[Text]
```

Return the parameters of the function `func` as a list of names.

