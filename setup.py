from setuptools import setup
setup(
  name = 'parsa',
  packages = ['parsa','parsa.backends','parsa.emulators'],
  package_dir = {'parsa': 'src'},
  version = '0.0.0.1',
  install_requires=[
        'pytest'
  ],
  description = "parsa: a natural language parser for bots",
  author = 'Alan Nichol',
  author_email = 'alan@golastmile.com',
  url = 'https://github.com/amn41/parsa'
)
