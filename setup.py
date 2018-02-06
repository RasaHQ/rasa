from setuptools import setup

__version__ = None  # Avoids IDE errors, but actual version is read from version.py
exec (open('rasa_nlu/version.py').read())

tests_requires = [
    "pytest",
    "pytest-pep8",
    "pytest-services",
    "pytest-cov",
    "pytest-twisted<1.6",
    "treq"
]

install_requires = [
    "pathlib",
    "cloudpickle",
    "gevent",
    "klein",
    "boto3",
    "typing",
    "future",
    "six",
    "tqdm",
    "requests",
    "jsonschema",
    "matplotlib",
    "numpy>=1.13",
    "simplejson",
]

extras_requires = {
    'test': tests_requires,
    'spacy': ["scikit-learn",
              "sklearn-crfsuite",
              "scipy",
              "spacy>2.0",
              ],
    'mitie': ["mitie"],
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
    include_package_data=True,
    description="Rasa NLU a natural language parser for bots",
    author='Alan Nichol',
    author_email='alan@rasa.ai',
    url="https://rasa.com",
    keywords=["nlp", "machine-learning", "machine-learning-library", "bot",
              "bots",
              "botkit", "rasa", "conversational-agents",
              "conversational-ai",
              "chatbot", "chatbot-framework", "bot-framework"],
    download_url="https://github.com/RasaHQ/rasa_nlu/archive/{}.tar.gz".format(__version__)
)

print("\nWelcome to Rasa NLU!")
print("If any questions please visit documentation page https://rasahq.github.io/rasa_nlu")
print("or join community chat on https://gitter.im/RasaHQ/rasa_nlu")
