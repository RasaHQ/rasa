# Rasa + Poetry = ♥️ 

1. Install Poetry with `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python`
2. Clone `rasa`, then do `cd rasa && git checkout poetry`
3. Install `pyenv` or any similar tool to manage environments. This information might be useful: https://python-poetry.org/docs/managing-environments/
4. Make sure you use Python 3.6+ within the current environment
5. Run `poetry install` to install Rasa in development mode

Optional:
- Enable autocompletions: https://python-poetry.org/docs/#enable-tab-completion-for-bash-fish-or-zsh

# Useful commands

Poetry won't really allow you to fuck the project up if you modify dependencies using 
these commands so feel free to experiment. It's better to use the commands instead of 
modifying the `pyproject.toml` manually.

#### Show all dependencies
`poetry show`

#### Show outdated dependencies
`poetry show --outdated`

#### Update the individual dependency to the latest version
`poetry update jsonschema@latest` (as an example)

#### Update it to any specific version
`poetry update jsonschema==3.2.0` (as an example)

#### Add new dev dependency
`poetry update jsonschema==3.2.0` (as an example)

#### Export dependencies
`poetry export -f requirements.txt > requirements.txt`

#### Export dev dependencies
`poetry export --dev -f requirements.txt > requirements-dev.txt`

### Other commands

https://python-poetry.org/docs/cli/#export