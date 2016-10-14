from setuptools import setup
setup(
  name = 'parsa',
  packages = ['parsa'],
  package_dir = {'parsa': 'src'},
  version = '0.0.0.1',
  install_requires=[
        'requests',
        'pytest'
        'flask',
        'mitie'
  ],
  dependency_links = ['git+https://github.com/mit-nlp/MITIE.git#egg=mitie'],
  description = "parsa: a natural language parser for bots",
  author = 'Alan Nichol',
  author_email = 'alan@golastmile.com',
  url = 'https://github.com/amn41/parsa'
)
