from setuptools import setup

__version__ = None   # Avoids IDE errors, but actual version is read from version.py
exec(open('rasa_nlu/version.py').read())

tests_requires = [
    "pytest-pep8",
    "pytest-xdist",
    "pytest-services",
    "pytest-flask",
]

install_requires = [
    "requests",
    "pathlib",
    "cloudpickle",
    "gevent",
    "flask",
    "boto3",
    "typing",
    "future",
    "six",
    "jsonschema"
]

extras_requires = {
    'test': tests_requires,
    'spacy': ["sklearn", "scipy", "numpy"],
    'mitie': ["mitie", "numpy"],
}

setup(
    name='rasa_nlu',
    packages=[
        'rasa_nlu',
        'rasa_nlu.utils',
        'rasa_nlu.classifiers',
        'rasa_nlu.emulators',
        'rasa_nlu.extractors',
        'rasa_nlu.featurizers',
        'rasa_nlu.tokenizers',
    ],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6"
    ],
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    description="rasa NLU a natural language parser for bots",
    author='Alan Nichol',
    author_email='alan@golastmile.com',
    url="https://rasa.ai",
    keywords=["NLP", "bots"],
    download_url="https://github.com/golastmile/rasa_nlu/archive/{}.tar.gz".format(__version__)
)
