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
        'rasa_nlu.tokenizers',
        'rasa_nlu.visualization'
    ],
    package_dir={'rasa_nlu': 'src'},
    version='0.4.2',
    install_requires=[],
    description="rasa NLU a natural language parser for bots",
    author='Alan Nichol',
    author_email='alan@golastmile.com',
    url="https://rasa.ai",
    keywords = ["NLP","bots"],
    download_url="https://github.com/golastmile/rasa_nlu/tarball/0.4.2"
)
