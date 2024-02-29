import asyncio
import functools
import importlib
import inspect
import logging
import pkgutil
import sys
from types import ModuleType
from typing import Text, Dict, Optional, Any, List, Callable, Collection, Type

import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_MIGRATION_GUIDE
from rasa.shared.exceptions import RasaException

logger = logging.getLogger(__name__)


def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Type:
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects.

    Args:
        module_path: either an absolute path to a Python class,
                     or the name of the class in the local / global scope.
        lookup_path: a path where to load the class from, if it cannot
                     be found in the local / global scope.

    Returns:
        a Python class

    Raises:
        ImportError, in case the Python class cannot be found.
        RasaException, in case the imported result is something other than a class
    """
    warn_and_exit_if_module_path_contains_rasa_plus(module_path, lookup_path)

    klass = None
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        klass = getattr(m, class_name, None)
    elif lookup_path:
        # try to import the class from the lookup path
        m = importlib.import_module(lookup_path)
        klass = getattr(m, module_path, None)

    if klass is None:
        raise ImportError(f"Cannot retrieve class from path {module_path}.")

    if not inspect.isclass(klass):
        raise RasaException(
            f"`class_from_module_path()` is expected to return a class, "
            f"but for {module_path} we got a {type(klass)}."
        )
    return klass


def import_package_modules(package_name: str) -> List[ModuleType]:
    """Import all modules in a package."""
    package = importlib.import_module(package_name)
    return [
        importlib.import_module(f"{package_name}.{module_name}")
        for _, module_name, _ in pkgutil.iter_modules(package.__path__)
    ]


def all_subclasses(cls: Any) -> List[Any]:
    """Returns all known (imported) subclasses of a class."""
    classes = cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in all_subclasses(s)
    ]

    return [subclass for subclass in classes if not inspect.isabstract(subclass)]


def module_path_from_instance(inst: Any) -> Text:
    """Return the module path of an instance's class."""
    return inst.__module__ + "." + inst.__class__.__name__


def sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]:
    """Sorts a list of dictionaries by their first key."""
    return sorted(dicts, key=lambda d: next(iter(d.keys())))


def lazy_property(function: Callable) -> Any:
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property.
    """
    attr_name = "_lazy_" + function.__name__

    def _lazyprop(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return property(_lazyprop)


def cached_method(f: Callable[..., Any]) -> Callable[..., Any]:
    """Caches method calls based on the call's `args` and `kwargs`.

    Works for `async` and `sync` methods. Don't apply this to functions.

    Args:
        f: The decorated method whose return value should be cached.

    Returns:
        The return value which the method gives for the first call with the given
        arguments.
    """
    assert "self" in arguments_of(f), "This decorator can only be used with methods."

    class Cache:
        """Helper class to abstract the caching details."""

        def __init__(self, caching_object: object, args: Any, kwargs: Any) -> None:
            self.caching_object = caching_object
            self.cache = getattr(caching_object, self._cache_name(), {})
            # noinspection PyUnresolvedReferences
            self.cache_key = functools._make_key(args, kwargs, typed=False)

        def _cache_name(self) -> Text:
            return f"_cached_{self.caching_object.__class__.__name__}_{f.__name__}"

        def is_cached(self) -> bool:
            return self.cache_key in self.cache

        def cache_result(self, result: Any) -> None:
            self.cache[self.cache_key] = result
            setattr(self.caching_object, self._cache_name(), self.cache)

        def cached_result(self) -> Any:
            return self.cache[self.cache_key]

    if asyncio.iscoroutinefunction(f):

        @functools.wraps(f)
        async def decorated(self: object, *args: Any, **kwargs: Any) -> Any:
            cache = Cache(self, args, kwargs)
            if not cache.is_cached():
                # Store the task immediately so that other concurrent calls of the
                # method can re-use the same task and don't schedule a second execution.
                to_cache = asyncio.ensure_future(f(self, *args, **kwargs))
                cache.cache_result(to_cache)
            return await cache.cached_result()

        return decorated
    else:

        @functools.wraps(f)
        def decorated(self: object, *args: Any, **kwargs: Any) -> Any:
            cache = Cache(self, args, kwargs)
            if not cache.is_cached():
                to_cache = f(self, *args, **kwargs)
                cache.cache_result(to_cache)
            return cache.cached_result()

        return decorated


def transform_collection_to_sentence(collection: Collection[Text]) -> Text:
    """Transforms e.g. a list like ['A', 'B', 'C'] into a sentence 'A, B and C'."""
    x = list(collection)
    if len(x) >= 2:
        return ", ".join(map(str, x[:-1])) + " and " + x[-1]
    return "".join(collection)


def minimal_kwargs(
    kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None
) -> Dict[Text, Any]:
    """Returns only the kwargs which are required by a function. Keys, contained in
    the exception list, are not included.

    Args:
        kwargs: All available kwargs.
        func: The function which should be called.
        excluded_keys: Keys to exclude from the result.

    Returns:
        Subset of kwargs which are accepted by `func`.

    """
    excluded_keys = excluded_keys or []

    possible_arguments = arguments_of(func)

    return {
        k: v
        for k, v in kwargs.items()
        if k in possible_arguments and k not in excluded_keys
    }


def mark_as_experimental_feature(feature_name: Text) -> None:
    """Warns users that they are using an experimental feature."""
    logger.warning(
        f"The {feature_name} is currently experimental and might change or be "
        "removed in the future 🔬 Please share your feedback on it in the "
        "forum (https://forum.rasa.com) to help us make this feature "
        "ready for production."
    )


def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())


def extract_duplicates(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Extracts duplicates from two lists."""
    if list1:
        dict1 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in list1
        }
    else:
        dict1 = {}

    if list2:
        dict2 = {
            (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in list2
        }
    else:
        dict2 = {}

    set1 = set(dict1.keys())
    set2 = set(dict2.keys())
    dupes = set1.intersection(set2)
    return sorted(list(dupes))


def clean_duplicates(dupes: Dict[Text, Any]) -> Dict[Text, Any]:
    """Removes keys for empty values."""
    duplicates = dupes.copy()
    for k in dupes:
        if not dupes[k]:
            duplicates.pop(k)

    return duplicates


def merge_dicts(
    tempDict1: Dict[Text, Any],
    tempDict2: Dict[Text, Any],
    override_existing_values: bool = False,
) -> Dict[Text, Any]:
    """Merges two dicts."""
    if override_existing_values:
        merged_dicts, b = tempDict1.copy(), tempDict2.copy()

    else:
        merged_dicts, b = tempDict2.copy(), tempDict1.copy()
    merged_dicts.update(b)
    return merged_dicts


def merge_lists(
    list1: List[Any], list2: List[Any], override: bool = False
) -> List[Any]:
    """Merges two lists."""
    return sorted(list(set(list1 + list2)))


def merge_lists_of_dicts(
    dict_list1: List[Dict],
    dict_list2: List[Dict],
    override_existing_values: bool = False,
) -> List[Dict]:
    """Merges two dict lists."""
    dict1 = {
        (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in dict_list1
    }
    dict2 = {
        (sorted(list(i.keys()))[0] if isinstance(i, dict) else i): i for i in dict_list2
    }
    merged_dicts = merge_dicts(dict1, dict2, override_existing_values)
    return list(merged_dicts.values())


def warn_and_exit_if_module_path_contains_rasa_plus(
    module_path: Text, lookup_path: Optional[str] = None
) -> None:
    """Warns and exits if the module path contains `rasa_plus`.

    Args:
        module_path: The module path to check.
        lookup_path: The lookup path to check.
    """
    if "rasa_plus" in module_path.lower() or (
        lookup_path and "rasa_plus" in lookup_path.lower()
    ):
        rasa.shared.utils.io.raise_warning(
            f"Your module path '{module_path}' contains 'rasa_plus'. "
            f"The path to this Rasa Pro component has changed, please "
            f"follow the migration guide in the official documentation "
            f"to update your code: ",
            docs=DOCS_URL_MIGRATION_GUIDE,
        )
        sys.exit(1)
