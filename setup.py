from setuptools import setup
setup(
  name = 'parsa',
  packages = [
    'parsa',
    'parsa.classifiers',
    'parsa.emulators',
    'parsa.extractors',
    'parsa.featurizers',
    'parsa.interpreters',
    'parsa.trainers',
    'parsa.tokenizers'
  ],
  package_dir = {'parsa': 'src'},
  version = '0.2.2',
  install_requires=[
        'pytest'
  ],
  description = "parsa: a natural language parser for bots",
  author = 'Alan Nichol',
  author_email = 'alan@golastmile.com',
  url = 'https://github.com/amn41/parsa'
)
