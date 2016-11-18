from setuptools import setup
setup(
  name = 'rasa_nlu',
  packages = [
    'rasa_nlu',
    'rasa_nlu.classifiers',
    'rasa_nlu.emulators',
    'rasa_nlu.extractors',
    'rasa_nlu.featurizers',
    'rasa_nlu.interpreters',
    'rasa_nlu.trainers',
    'rasa_nlu.tokenizers'
  ],
  package_dir = {'rasa_nlu': 'src'},
  version = '0.3.0',
  install_requires=[
        'pytest'
  ],
  extras_require = {
    'spacy': ["spacy"],
    'mitie': ["mitie"],
    'sklearn': ["scikit-learn"]
  },
  dependency_links = ["git+https://github.com/mit-nlp/MITIE.git#egg=mitie"],
  description = "rasa_nlu: a natural language parser for bots",
  author = 'Alan Nichol',
  author_email = 'alan@golastmile.com'
)
