---
sidebar_label: rasa.shared.core.training_data.story_reader.markdown_story_reader
title: rasa.shared.core.training_data.story_reader.markdown_story_reader
---
## MarkdownStoryReader Objects

```python
class MarkdownStoryReader(StoryReader)
```

Class that reads the core training data in a Markdown format.

#### \_\_init\_\_

```python
 | __init__(domain: Optional[Domain] = None, template_vars: Optional[Dict] = None, use_e2e: bool = False, source_name: Optional[Text] = None, is_used_for_training: bool = True, ignore_deprecation_warning: bool = False) -> None
```

Creates reader. See parent class docstring for more information.

#### read\_from\_file

```python
 | read_from_file(filename: Union[Text, Path], skip_validation: bool = False) -> List[StoryStep]
```

Given a md file reads the contained stories.

#### parse\_e2e\_message

```python
 | parse_e2e_message(line: Text, is_used_for_training: bool = True) -> Message
```

Parses an md list item line based on the current section type.

Matches expressions of the form `&lt;intent&gt;:&lt;example&gt;`. For the
syntax of `&lt;example&gt;` see the Rasa docs on NLU training data.

#### is\_stories\_file

```python
 | @staticmethod
 | is_stories_file(file_path: Union[Text, Path]) -> bool
```

Check if file contains Core training data or rule data in Markdown format.

**Arguments**:

- `file_path` - Path of the file to check.
  

**Returns**:

  `True` in case the file is a Core Markdown training data or rule data file,
  `False` otherwise.

#### is\_test\_stories\_file

```python
 | @staticmethod
 | is_test_stories_file(file_path: Union[Text, Path]) -> bool
```

Checks if a file contains test stories.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a file containing test stories, otherwise `False`.

