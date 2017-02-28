from setuptools import setup

__version__ = None   # Avoids IDE errors, but actual version is read from version.py
exec(open('src/version.py').read())

setup(
    name='rasa_nlu',
    packages=[
        'rasa_nlu',
        'rasa_nlu.utils',
        'rasa_nlu.classifiers',
        'rasa_nlu.emulators',
        'rasa_nlu.extractors',
        'rasa_nlu.featurizers',
        'rasa_nlu.interpreters',
        'rasa_nlu.trainers',
        'rasa_nlu.tokenizers'
    ],
    package_dir={'rasa_nlu': 'src'},
    version=__version__,
    install_requires=[],
    description="rasa NLU a natural language parser for bots",
    author='Alan Nichol',
    author_email='alan@golastmile.com',
    url="https://rasa.ai",
    keywords=["NLP", "bots"],
    download_url="https://github.com/golastmile/rasa_nlu/tarball/0.6-beta"
)
