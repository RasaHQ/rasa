from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('rasa_core/version.py').read())

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

tests_requires = [
    "pytest",
    "pytest-pep8",
    "pytest-services",
    "pytest-cov",
    "pytest-xdist",
    "pytest-twisted<1.6",
    "treq",
    "freezegun",
]

install_requires = [
    'jsonpickle',
    'six',
    'redis',
    'fakeredis',
    'nbsphinx',
    'pandoc',
    'future',
    'numpy>=1.13',
    'typing>=3.6',
    'requests',
    'graphviz',
    'Keras',
    'tensorflow',
    'h5py',
    'apscheduler',
    'tqdm',
    'ConfigArgParse',
    'networkx',
    'fbmessenger>=5.0.0',
    'pykwalify<=1.6.0',
    'coloredlogs',
    'ruamel.yaml',
    'flask',
    'scikit-learn',
    'rasa_nlu>=0.12.0',
    'slackclient',
    'python-telegram-bot',
    'twilio',
    'mattermostwrapper'
]

extras_requires = {
    'test': tests_requires
}

setup(
    name='rasa-core',
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
    packages=find_packages(exclude=["tests", "tools"]),
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    include_package_data=True,
    description="Machine learning based dialogue engine "
                "for conversational software.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Rasa Technologies GmbH',
    author_email='hi@rasa.com',
    maintainer="Tom Bocklisch",
    maintainer_email="tom@rasa.com",
    license='Apache 2.0',
    keywords="nlp machine-learning machine-learning-library bot bots "
             "botkit rasa conversational-agents conversational-ai chatbot"
             "chatbot-framework bot-framework",
    url="https://rasa.ai",
    download_url="https://github.com/RasaHQ/rasa_core/archive/{}.tar.gz".format(__version__),
    project_urls={
        'Bug Reports': 'https://github.com/rasahq/rasa_core/issues',
        'Source': 'https://github.com/rasahq/rasa_core',
    },
)

print("\nWelcome to Rasa Core!")
print("If any questions please visit documentation page https://core.rasa.com")
print("or join community chat on https://gitter.im/RasaHQ/rasa_core")
