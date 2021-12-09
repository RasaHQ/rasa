---
sidebar_label: rasa.core.training.story_reader.markdown_story_reader
title: rasa.core.training.story_reader.markdown_story_reader
---
## MarkdownStoryReader Objects

```python
class MarkdownStoryReader(StoryReader)
```

Class that reads the core training data in a Markdown format

#### read\_from\_file

```python
 | async read_from_file(filename: Union[Text, Path]) -> List[StoryStep]
```

Given a md file reads the contained stories.

#### parse\_e2e\_message

```python
 | @staticmethod
 | parse_e2e_message(line: Text) -> Message
```

Parses an md list item line based on the current section type.

Matches expressions of the form `&lt;intent&gt;:&lt;example&gt;`. For the
syntax of `&lt;example&gt;` see the Rasa docs on NLU training data.

#### is\_markdown\_story\_file

```python
 | @staticmethod
 | is_markdown_story_file(file_path: Union[Text, Path]) -> bool
```

Check if file contains Core training data or rule data in Markdown format.

**Arguments**:

- `file_path` - Path of the file to check.
  

**Returns**:

  `True` in case the file is a Core Markdown training data or rule data file,
  `False` otherwise.

#### is\_markdown\_test\_stories\_file

```python
 | @staticmethod
 | is_markdown_test_stories_file(file_path: Union[Text, Path]) -> bool
```

Checks if a file contains test stories.

**Arguments**:

- `file_path` - Path of the file which should be checked.
  

**Returns**:

  `True` if it&#x27;s a file containing test stories, otherwise `False`.

