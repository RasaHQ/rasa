import asyncio
import functools
import importlib
import inspect
import logging
from typing import Text, Dict, Optional, Any, List, Callable, Collection

import rasa.shared.utils.io
from rasa.shared.constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS


logger = logging.getLogger(__name__)


def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Any:
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
    """
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
        rasa.shared.utils.io.raise_deprecation_warning(
            f"`class_from_module_path()` is expected to return a class, "
            f"but {module_path} is not one. "
            f"This warning will be converted "
            f"into an exception in {NEXT_MAJOR_VERSION_FOR_DEPRECATIONS}."
        )

    return klass


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
    return sorted(dicts, key=lambda d: list(d.keys())[0])


def lazy_property(function: Callable) -> Any:
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property."""

    attr_name = "_lazy_" + function.__name__

    @property
    def _lazyprop(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return _lazyprop


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
