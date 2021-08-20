---
sidebar_label: rasa.nlu.registry
title: rasa.nlu.registry
---
This is a somewhat delicate package. It contains all registered components
and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should
import this in module scope.

## ComponentNotFoundException Objects

```python
class ComponentNotFoundException(ModuleNotFoundError,  RasaException)
```

Raised if a module referenced by name can not be imported.

#### get\_component\_class

```python
get_component_class(component_name: Text) -> Type["Component"]
```

Resolve component name to a registered components class.

#### load\_component\_by\_meta

```python
load_component_by_meta(component_meta: Dict[Text, Any], model_dir: Text, metadata: Metadata, cached_component: Optional["Component"], **kwargs: Any, ,) -> Optional["Component"]
```

Resolves a component and calls its load method.

Inits it based on a previously persisted model.

#### create\_component\_by\_config

```python
create_component_by_config(component_config: Dict[Text, Any], config: "RasaNLUModelConfig") -> Optional["Component"]
```

Resolves a component and calls it&#x27;s create method.

Inits it based on a previously persisted model.

