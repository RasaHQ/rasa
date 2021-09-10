---
sidebar_label: rasa.shared.utils.io
title: rasa.shared.utils.io
---
#### raise\_warning

```python
def raise_warning(message: Text, category: Optional[Type[Warning]] = None, docs: Optional[Text] = None, **kwargs: Any, ,) -> None
```

Emit a `warnings.warn` with sensible defaults and a colored warning msg.

#### write\_text\_file

```python
def write_text_file(content: Text, file_path: Union[Text, Path], encoding: Text = DEFAULT_ENCODING, append: bool = False) -> None
```

Writes text to a file.

**Arguments**:

- `content` - The content to write.
- `file_path` - The path to which the content should be written.
- `encoding` - The encoding which should be used.
- `append` - Whether to append to the file or to truncate the file.

#### read\_file

```python
def read_file(filename: Union[Text, Path], encoding: Text = DEFAULT_ENCODING) -> Any
```

Read text from a file.

#### read\_json\_file

```python
def read_json_file(filename: Union[Text, Path]) -> Any
```

Read json from a file.

#### list\_directory

```python
def list_directory(path: Text) -> List[Text]
```

Returns all files and folders excluding hidden files.

If the path points to a file, returns the file. This is a recursive
implementation returning files in any depth of the path.

#### list\_files

```python
def list_files(path: Text) -> List[Text]
```

Returns all files excluding hidden files.

If the path points to a file, returns the file.

#### list\_subdirectories

```python
def list_subdirectories(path: Text) -> List[Text]
```

Returns all folders excluding hidden files.

If the path points to a file, returns an empty list.

#### deep\_container\_fingerprint

```python
def deep_container_fingerprint(obj: Union[List[Any], Dict[Any, Any], Any], encoding: Text = DEFAULT_ENCODING) -> Text
```

Calculate a hash which is stable, independent of a containers key order.

Works for lists and dictionaries. For keys and values, we recursively call
`hash(...)` on them. Keep in mind that a list with keys in a different order
will create the same hash!

**Arguments**:

- `obj` - dictionary or list to be hashed.
- `encoding` - encoding used for dumping objects as strings
  

**Returns**:

  hash of the container.

#### get\_dictionary\_fingerprint

```python
def get_dictionary_fingerprint(dictionary: Dict[Any, Any], encoding: Text = DEFAULT_ENCODING) -> Text
```

Calculate the fingerprint for a dictionary.

The dictionary can contain any keys and values which are either a dict,
a list or a elements which can be dumped as a string.

**Arguments**:

- `dictionary` - dictionary to be hashed
- `encoding` - encoding used for dumping objects as strings
  

**Returns**:

  The hash of the dictionary

#### get\_list\_fingerprint

```python
def get_list_fingerprint(elements: List[Any], encoding: Text = DEFAULT_ENCODING) -> Text
```

Calculate a fingerprint for an unordered list.

**Arguments**:

- `elements` - unordered list
- `encoding` - encoding used for dumping objects as strings
  

**Returns**:

  the fingerprint of the list

#### get\_text\_hash

```python
def get_text_hash(text: Text, encoding: Text = DEFAULT_ENCODING) -> Text
```

Calculate the md5 hash for a text.

#### json\_to\_string

```python
def json_to_string(obj: Any, **kwargs: Any) -> Text
```

Dumps a JSON-serializable object to string.

**Arguments**:

- `obj` - JSON-serializable object.
- `kwargs` - serialization options. Defaults to 2 space indentation
  and disable escaping of non-ASCII characters.
  

**Returns**:

  The objects serialized to JSON, as a string.

#### fix\_yaml\_loader

```python
def fix_yaml_loader() -> None
```

Ensure that any string read by yaml is represented as unicode.

#### replace\_environment\_variables

```python
def replace_environment_variables() -> None
```

Enable yaml loader to process the environment variables in the yaml.

#### read\_yaml

```python
def read_yaml(content: Text, reader_type: Union[Text, List[Text]] = "safe") -> Any
```

Parses yaml from a text.

**Arguments**:

- `content` - A text containing yaml content.
- `reader_type` - Reader type to use. By default &quot;safe&quot; will be used.
  

**Raises**:

- `ruamel.yaml.parser.ParserError` - If there was an error when parsing the YAML.

#### read\_yaml\_file

```python
def read_yaml_file(filename: Union[Text, Path]) -> Union[List[Any], Dict[Text, Any]]
```

Parses a yaml file.

Raises an exception if the content of the file can not be parsed as YAML.

**Arguments**:

- `filename` - The path to the file which should be read.
  

**Returns**:

  Parsed content of the file.

#### write\_yaml

```python
def write_yaml(data: Any, target: Union[Text, Path, StringIO], should_preserve_key_order: bool = False) -> None
```

Writes a yaml to the file or to the stream

**Arguments**:

- `data` - The data to write.
- `target` - The path to the file which should be written or a stream object
- `should_preserve_key_order` - Whether to force preserve key order in `data`.

#### is\_key\_in\_yaml

```python
def is_key_in_yaml(file_path: Union[Text, Path], *keys: Text) -> bool
```

Checks if any of the keys is contained in the root object of the yaml file.

**Arguments**:

- `file_path` - path to the yaml file
- `keys` - keys to look for
  

**Returns**:

  `True` if at least one of the keys is found, `False` otherwise.
  

**Raises**:

- `FileNotFoundException` - if the file cannot be found.

#### convert\_to\_ordered\_dict

```python
def convert_to_ordered_dict(obj: Any) -> Any
```

Convert object to an `OrderedDict`.

**Arguments**:

- `obj` - Object to convert.
  

**Returns**:

  An `OrderedDict` with all nested dictionaries converted if `obj` is a
  dictionary, otherwise the object itself.

#### is\_logging\_disabled

```python
def is_logging_disabled() -> bool
```

Returns `True` if log level is set to WARNING or ERROR, `False` otherwise.

#### create\_directory\_for\_file

```python
def create_directory_for_file(file_path: Union[Text, Path]) -> None
```

Creates any missing parent directories of this file path.

#### dump\_obj\_as\_json\_to\_file

```python
def dump_obj_as_json_to_file(filename: Union[Text, Path], obj: Any) -> None
```

Dump an object as a json string to a file.

#### dump\_obj\_as\_yaml\_to\_string

```python
def dump_obj_as_yaml_to_string(obj: Any, should_preserve_key_order: bool = False) -> Text
```

Writes data (python dict) to a yaml string.

**Arguments**:

- `obj` - The object to dump. Has to be serializable.
- `should_preserve_key_order` - Whether to force preserve key order in `data`.
  

**Returns**:

  The object converted to a YAML string.

#### create\_directory

```python
def create_directory(directory_path: Text) -> None
```

Creates a directory and its super paths.

Succeeds even if the path already exists.

#### raise\_deprecation\_warning

```python
def raise_deprecation_warning(message: Text, warn_until_version: Text = NEXT_MAJOR_VERSION_FOR_DEPRECATIONS, docs: Optional[Text] = None, **kwargs: Any, ,) -> None
```

Thin wrapper around `raise_warning()` to raise a deprecation warning. It requires
a version until which we&#x27;ll warn, and after which the support for the feature will
be removed.

#### read\_validated\_yaml

```python
def read_validated_yaml(filename: Union[Text, Path], schema: Text) -> Any
```

Validates YAML file content and returns parsed content.

**Arguments**:

- `filename` - The path to the file which should be read.
- `schema` - The path to the schema file which should be used for validating the
  file content.
  

**Returns**:

  The parsed file content.
  

**Raises**:

- `YamlValidationException` - In case the model configuration doesn&#x27;t match the
  expected schema.

#### read\_config\_file

```python
def read_config_file(filename: Union[Path, Text]) -> Dict[Text, Any]
```

Parses a yaml configuration file. Content needs to be a dictionary.

**Arguments**:

- `filename` - The path to the file which should be read.
  

**Raises**:

- `YamlValidationException` - In case file content is not a `Dict`.
  

**Returns**:

  Parsed config file.

#### read\_model\_configuration

```python
def read_model_configuration(filename: Union[Path, Text]) -> Dict[Text, Any]
```

Parses a model configuration file.

**Arguments**:

- `filename` - The path to the file which should be read.
  

**Raises**:

- `YamlValidationException` - In case the model configuration doesn&#x27;t match the
  expected schema.
  

**Returns**:

  Parsed config file.

#### is\_subdirectory

```python
def is_subdirectory(path: Text, potential_parent_directory: Text) -> bool
```

Checks if `path` is a subdirectory of `potential_parent_directory`.

**Arguments**:

- `path` - Path to a file or directory.
- `potential_parent_directory` - Potential parent directory.
  

**Returns**:

  `True` if `path` is a subdirectory of `potential_parent_directory`.

#### random\_string

```python
def random_string(length: int) -> Text
```

Returns a random string of given length.

