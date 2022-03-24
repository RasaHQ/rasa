from yamllint.cli import run

from yamllint.cli import run
def lint_file(path: str) -> None:
    #conf = YamlLintConfig('extends: default')
    #gen = linter.run(f, conf)
    run({path})