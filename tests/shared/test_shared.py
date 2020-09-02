import os
from pathlib import Path


def test_shared_package_is_independent():
    shared_package = Path(".") / "rasa" / "shared"

    for root, dirs, files in os.walk(shared_package):
        python_files = [f for f in files if f.endswith(".py")]

        for file in python_files:
            full_path = Path(root) / file
            lines = full_path.read_text().split("\n")
            lines = [line.strip() for line in lines]

            imports = [
                line
                for line in lines
                if line.startswith("import ") or line.startswith("from ")
            ]
            rasa_imports = [line for line in imports if "rasa" in line]

            outside_rasa_imports = [
                import_line
                for import_line in rasa_imports
                if "rasa.shared" not in import_line
            ]

            # The shared package is required to be independent of the rest of Rasa
            assert not outside_rasa_imports
