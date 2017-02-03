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
    version='0.6-beta',
    install_requires=[],
    description="rasa NLU a natural language parser for bots",
    author='Alan Nichol',
    author_email='alan@golastmile.com',
    url="https://rasa.ai",
    keywords=["NLP", "bots"],
    download_url="https://github.com/golastmile/rasa_nlu/tarball/0.6-beta"
)
