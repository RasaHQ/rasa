import io
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('rasa_nlu/version.py').read())

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

tests_requires = [
    "pytest",
    "pytest-pep8",
    "pytest-cov",
    "pytest-twisted<1.6",
    "treq",
    "responses~=0.9.0",
    "httpretty~=0.9.0",
]

install_requires = [
    "pathlib",
    "cloudpickle",
    "gevent",
    "klein",
    "boto3",
    "packaging",
    "typing",
    "future",
    "six",
    "tqdm",
    "requests",
    "jsonschema",
    "matplotlib~=2.0",
    "numpy>=1.13",
    "simplejson",
    "pyyaml",
    'coloredlogs',
]

extras_requires = {
    'test': tests_requires,
    'spacy': ["scikit-learn<0.20",
              "sklearn-crfsuite",
              "scipy",
              "spacy>2.0",
              ],
    'tensorflow': ["scikit-learn<0.20",
                   "sklearn-crfsuite",
                   "scipy",
                   "tensorflow"
                   ],
    'mitie': ["mitie"],
}

setup(
    name='rasa-nlu',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        # supported python versions
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
    ],
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    include_package_data=True,
    description="Rasa NLU a natural language parser for bots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Rasa Technologies GmbH',
    author_email='hi@rasa.com',
    maintainer="Tom Bocklisch",
    maintainer_email="tom@rasa.com",
    license='Apache 2.0',
    url="https://rasa.com",
    keywords="nlp machine-learning machine-learning-library bot bots "
             "botkit rasa conversational-agents conversational-ai chatbot"
             "chatbot-framework bot-framework",
    download_url="https://github.com/RasaHQ/rasa_nlu/archive/{}.tar.gz"
                 "".format(__version__),
    project_urls={
        'Bug Reports': 'https://github.com/rasahq/rasa_nlu/issues',
        'Source': 'https://github.com/rasahq/rasa_nlu',
    },
)

print("\nWelcome to Rasa NLU!")
print("If any questions please visit documentation "
      "page https://rasa.com/docs/nlu/")
print("or join the community discussions on https://forum.rasa.com")
