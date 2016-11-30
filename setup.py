from setuptools import setup

setup(
    name='rasa_nlu',
    packages=[
        'rasa_nlu',
        'rasa_nlu.classifiers',
        'rasa_nlu.emulators',
        'rasa_nlu.extractors',
        'rasa_nlu.featurizers',
        'rasa_nlu.interpreters',
        'rasa_nlu.trainers',
        'rasa_nlu.tokenizers'
    ],
    package_dir={'rasa_nlu': 'src'},
    version='0.4.0',
    install_requires=[],
    description="rasa_nlu: a natural language parser for bots",
    author='Alan Nichol',
    author_email='alan@golastmile.com',
    url="http://rasa.ai"
)
