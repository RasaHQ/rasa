import importlib
import pkgutil

import pytest
from six import PY2


def import_submodules(package):
    """ Import all submodules of a module, recursively, including subpackages."""

    if isinstance(package, str):
        package = importlib.import_module(package)
    results = []
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        module = importlib.import_module(full_name)
        if PY2:
            reload(module)
        else:
            importlib.reload(module)
        results.append(full_name)
        if is_pkg:
            results += import_submodules(full_name)
    return results


@pytest.mark.parametrize("banned_package", ["spacy", "mitie"])
def test_no_global_imports_of_banned_package(banned_package):
    """This test ensures that neither spacy nor mitie are imported module wise in any of our code files.

    If one of the dependencies is needed, they should be imported within a function."""
    import inspect

    # To track imports accross modules, we will replace the default import function
    try:
        import __builtin__
        savimp = __builtin__.__import__
    except ImportError:
        import builtins
        savimp = builtins.__import__

    tracked_imports = dict()

    def import_tracking(name, *x, **xs):
        caller = inspect.currentframe().f_back
        caller_name = caller.f_globals.get('__name__')
        if name in tracked_imports:
            tracked_imports[name].append(caller_name)
        else:
            tracked_imports[name] = [caller_name]
        return savimp(name, *x, **xs)

    if PY2:
        __builtin__.__import__ = import_tracking
    else:
        builtins.__import__ = import_tracking

    # import all available modules and track imports on the way
    import_submodules("rasa_nlu")

    def find_modules_importing(name):
        return [v for k, vs in tracked_imports.items() if k.startswith(name) for v in vs]

    assert not find_modules_importing(banned_package), \
        "No module should import {} globally. Found in {}".format(
            banned_package, ", ".join(find_modules_importing("spacy")))
