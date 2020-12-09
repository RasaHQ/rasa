import re
from typing import Dict, List, Text, Union, Optional

import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingDataFull


def _convert_lookup_tables_to_regex(
    training_data: TrainingDataFull, pattern_names: Optional[List[Text]] = None
) -> List[Dict[Text, Text]]:
    """Convert the lookup tables from the training data to regex patterns.

    Args:
        training_data: The training data.
        pattern_names: List of pattern names to use. If list is empty or `None` all
          patterns will be used.

    Returns:
        A list of regex patterns.
    """
    patterns = []
    for table in training_data.lookup_tables:
        if pattern_names and table["name"] not in pattern_names:
            continue
        regex_pattern = _generate_lookup_regex(table)
        lookup_regex = {"name": table["name"], "pattern": regex_pattern}
        patterns.append(lookup_regex)
    return patterns


def _generate_lookup_regex(lookup_table: Dict[Text, Union[Text, List[Text]]]) -> Text:
    """Creates a regex pattern from the given lookup table.

    The lookup table is either a file or a list of entries.

    Args:
        lookup_table: The lookup table.

    Returns:
        The regex pattern.
    """
    lookup_elements = lookup_table["elements"]

    # if it's a list, it should be the elements directly
    if isinstance(lookup_elements, list):
        elements_to_regex = lookup_elements
    # otherwise it's a file path.
    else:
        elements_to_regex = read_lookup_table_file(lookup_elements)

    # sanitize the regex, escape special characters
    elements_sanitized = [re.escape(e) for e in elements_to_regex]

    # regex matching elements with word boundaries on either side
    return "(\\b" + "\\b|\\b".join(elements_sanitized) + "\\b)"


def read_lookup_table_file(lookup_table_file: Text) -> List[Text]:
    """Read the lookup table file.

    Args:
        lookup_table_file: the file path to the lookup table

    Returns:
        Elements listed in the lookup table file.
    """
    try:
        f = open(lookup_table_file, "r", encoding=rasa.shared.utils.io.DEFAULT_ENCODING)
    except OSError:
        raise ValueError(
            f"Could not load lookup table {lookup_table_file}. "
            f"Please make sure you've provided the correct path."
        )

    elements_to_regex = []
    with f:
        for line in f:
            new_element = line.strip()
            if new_element:
                elements_to_regex.append(new_element)
    return elements_to_regex


def _collect_regex_features(
    training_data: TrainingDataFull, pattern_names: Optional[List[Text]] = None
) -> List[Dict[Text, Text]]:
    """Get regex features from training data.

    Args:
        training_data: The training data
        pattern_names: List of pattern names to use. If list is empty or None all
          patterns will be used.

    Returns:
        Regex features.
    """
    if not pattern_names:
        return training_data.regex_features

    return [
        regex
        for regex in training_data.regex_features
        if regex["name"] in pattern_names
    ]


def extract_patterns(
    training_data: TrainingDataFull,
    use_lookup_tables: bool = True,
    use_regexes: bool = True,
    patter_names: Optional[List[Text]] = None,
) -> List[Dict[Text, Text]]:
    """Extract a list of patterns from the training data.

    The patterns are constructed using the regex features and lookup tables defined
    in the training data.

    Args:
        training_data: The training data.
        patter_names: List of pattern names to use. If list is empty or `None` all
          patterns will be used.
        use_regexes: Boolean indicating whether to use regex features or not.
        use_lookup_tables: Boolean indicating whether to use lookup tables or not.

    Returns:
        The list of regex patterns.
    """
    if not training_data.lookup_tables and not training_data.regex_features:
        return []

    patterns = []

    if use_regexes:
        patterns.extend(_collect_regex_features(training_data, patter_names))
    if use_lookup_tables:
        patterns.extend(_convert_lookup_tables_to_regex(training_data, patter_names))

    return patterns
