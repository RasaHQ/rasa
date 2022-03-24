import argparse
import os
from rasa.shared.utils.cli import print_error

from yamllint.cli import run
def lint_file(path: str) -> None:
    #conf = YamlLintConfig('extends: default')
    #gen = linter.run(f, conf)
    os.system(f'yamllint {path}')