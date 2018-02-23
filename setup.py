from setuptools import setup, find_packages
import io

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('rasa_nlu/version.py').read())

try:
    import pypandoc
    readme = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError):
    with io.open('README.md', encoding='utf-8') as f:
        readme = f.read()

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
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        # supported python versions
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
    long_description=readme,
    author='Rasa Technologies GmbH',
    author_email='hi@rasa.ai',
    license='Apache 2.0',
    url="https://rasa.com",
    keywords="nlp machine-learning machine-learning-library bot bots "
             "botkit rasa conversational-agents conversational-ai chatbot"
             "chatbot-framework bot-framework",
    download_url="https://github.com/RasaHQ/rasa_nlu/archive/{}.tar.gz"
                 "".format(__version__)
)

print("\nWelcome to Rasa NLU!")
print("If any questions please visit documentation "
      "page https://rasahq.github.io/rasa_nlu")
print("or join community chat on https://gitter.im/RasaHQ/rasa_nlu")
