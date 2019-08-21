from typing import List, Set, Text


class PackageManager:
    @staticmethod
    def find_unavailable_packages(package_names: List[Text]) -> Set[Text]:
        """Tries to import all the package names and returns
        the packages where it failed."""
        import importlib

        failed_imports = set()
        for package in package_names:
            try:
                importlib.import_module(package)
            except ImportError:
                failed_imports.add(package)
        return failed_imports

    @staticmethod
    def validate_requirements(component_names: List[Text]) -> None:
        """Ensures that all required importable python packages are installed to
        instantiate and used the passed components."""
        from rasa.nlu import registry

        # Validate that all required packages are installed
        failed_imports = set()
        for component_name in component_names:
            component_class = registry.get_component_class(component_name)
            failed_imports.update(
                PackageManager.find_unavailable_packages(component_class.required_packages())
            )
        if failed_imports:  # pragma: no cover
            # if available, use the development file to figure out the correct
            # version numbers for each requirement
            raise Exception(
                "Not all required importable packages are installed. "
                + "To use this pipeline, you need to install the "
                  "missing dependencies. "
                + "Please install the package(s) that contain the module(s): {}".format(
                    ", ".join(failed_imports)
                )
            )