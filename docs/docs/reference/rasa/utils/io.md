---
sidebar_label: rasa.utils.io
title: rasa.utils.io
---

#### fix\_yaml\_loader

```python
fix_yaml_loader() -> None
```

Ensure that any string read by yaml is represented as unicode.

#### replace\_environment\_variables

```python
replace_environment_variables() -> None
```

Enable yaml loader to process the environment variables in the yaml.

#### read\_yaml

```python
read_yaml(content: Text) -> Any
```

Parses yaml from a text.

**Arguments**:

- `content` - A text containing yaml content.
  

**Raises**:

- `ruamel.yaml.parser.ParserError` - If there was an error when parsing the YAML.

#### read\_file

```python
read_file(filename: Union[Text, Path], encoding: Text = DEFAULT_ENCODING) -> Any
```

Read text from a file.

#### read\_json\_file

```python
read_json_file(filename: Union[Text, Path]) -> Any
```

Read json from a file.

#### dump\_obj\_as\_json\_to\_file

```python
dump_obj_as_json_to_file(filename: Union[Text, Path], obj: Any) -> None
```

Dump an object as a json string to a file.

#### pickle\_dump

```python
pickle_dump(filename: Union[Text, Path], obj: Any) -> None
```

Saves object to file.

**Arguments**:

- `filename` - the filename to save the object to
- `obj` - the object to store

#### pickle\_load

```python
pickle_load(filename: Union[Text, Path]) -> Any
```

Loads an object from a file.

**Arguments**:

- `filename` - the filename to load the object from
  
- `Returns` - the loaded object

#### read\_config\_file

```python
read_config_file(filename: Text) -> Dict[Text, Any]
```

Parses a yaml configuration file. Content needs to be a dictionary

**Arguments**:

- `filename` - The path to the file which should be read.

#### read\_yaml\_file

```python
read_yaml_file(filename: Union[Text, Path]) -> Union[List[Any], Dict[Text, Any]]
```

Parses a yaml file.

**Arguments**:

- `filename` - The path to the file which should be read.

#### unarchive

```python
unarchive(byte_array: bytes, directory: Text) -> Text
```

Tries to unpack a byte array interpreting it as an archive.

Tries to use tar first to unpack, if that fails, zip will be used.

#### convert\_to\_ordered\_dict

```python
convert_to_ordered_dict(obj: Any) -> Any
```

Convert object to an `OrderedDict`.

**Arguments**:

- `obj` - Object to convert.
  

**Returns**:

  An `OrderedDict` with all nested dictionaries converted if `obj` is a
  dictionary, otherwise the object itself.

#### write\_yaml

```python
write_yaml(data: Any, target: Union[Text, Path, StringIO], should_preserve_key_order: bool = False) -> None
```

Writes a yaml to the file or to the stream

**Arguments**:

- `data` - The data to write.
- `target` - The path to the file which should be written or a stream object
- `should_preserve_key_order` - Whether to force preserve key order in `data`.

#### write\_text\_file

```python
write_text_file(content: Text, file_path: Union[Text, Path], encoding: Text = DEFAULT_ENCODING, append: bool = False) -> None
```

Writes text to a file.

**Arguments**:

- `content` - The content to write.
- `file_path` - The path to which the content should be written.
- `encoding` - The encoding which should be used.
- `append` - Whether to append to the file or to truncate the file.

#### create\_temporary\_file

```python
create_temporary_file(data: Any, suffix: Text = "", mode: Text = "w+") -> Text
```

Creates a tempfile.NamedTemporaryFile object for data.

mode defines NamedTemporaryFile&#x27;s  mode parameter in py3.

#### create\_temporary\_directory

```python
create_temporary_directory() -> Text
```

Creates a tempfile.TemporaryDirectory.

#### create\_path

```python
create_path(file_path: Text) -> None
```

Makes sure all directories in the &#x27;file_path&#x27; exists.

#### create\_directory\_for\_file

```python
create_directory_for_file(file_path: Union[Text, Path]) -> None
```

Creates any missing parent directories of this file path.

#### file\_type\_validator

```python
file_type_validator(valid_file_types: List[Text], error_message: Text) -> Type["Validator"]
```

Creates a `Validator` class which can be used with `questionary` to validate
file paths.

#### not\_empty\_validator

```python
not_empty_validator(error_message: Text) -> Type["Validator"]
```

Creates a `Validator` class which can be used with `questionary` to validate
that the user entered something other than whitespace.

#### create\_validator

```python
create_validator(function: Callable[[Text], bool], error_message: Text) -> Type["Validator"]
```

Helper method to create `Validator` classes from callable functions. Should be
removed when questionary supports `Validator` objects.

#### list\_files

```python
list_files(path: Text) -> List[Text]
```

Returns all files excluding hidden files.

If the path points to a file, returns the file.

#### list\_subdirectories

```python
list_subdirectories(path: Text) -> List[Text]
```

Returns all folders excluding hidden files.

If the path points to a file, returns an empty list.

#### list\_directory

```python
list_directory(path: Text) -> List[Text]
```

Returns all files and folders excluding hidden files.

If the path points to a file, returns the file. This is a recursive
implementation returning files in any depth of the path.

#### create\_directory

```python
create_directory(directory_path: Text) -> None
```

Creates a directory and its super paths.

Succeeds even if the path already exists.

#### zip\_folder

```python
zip_folder(folder: Text) -> Text
```

Create an archive from a folder.

#### json\_unpickle

```python
json_unpickle(file_name: Union[Text, Path]) -> Any
```

Unpickle an object from file using json.

**Arguments**:

- `file_name` - the file to load the object from
  
- `Returns` - the object

#### json\_pickle

```python
json_pickle(file_name: Union[Text, Path], obj: Any) -> None
```

Pickle an object to a file using json.

**Arguments**:

- `file_name` - the file to store the object to
- `obj` - the object to store

#### encode\_string

```python
encode_string(s: Text) -> Text
```

Return an encoded python string.

#### decode\_string

```python
decode_string(s: Text) -> Text
```

Return a decoded python string.

