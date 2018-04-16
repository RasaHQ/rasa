from setuptools import setup, find_packages
import io

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('rasa_core/version.py').read())

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
    'rasa_nlu>=0.12.0a2',
    'slackclient',
    'python-telegram-bot',
    'twilio',
    'mattermostwrapper'
]

extras_requires = {
    'test': tests_requires
}

setup(
    name='rasa_core',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        # supported python versions
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6"
    ],
    packages=find_packages(exclude=["tests", "tools"]),
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    include_package_data=True,
    description="Machine learning based dialogue engine "
                "for conversational software.",
    long_description=readme,
    author='Rasa Technologies GmbH',
    author_email='hi@rasa.com',
    license='Apache 2.0',
    keywords="nlp machine-learning machine-learning-library bot bots "
             "botkit rasa conversational-agents conversational-ai chatbot"
             "chatbot-framework bot-framework",
    url="https://rasa.ai",
    download_url="https://github.com/RasaHQ/rasa_core/archive/{}.tar.gz".format(__version__)
)

print("\nWelcome to Rasa Core!")
print("If any questions please visit documentation page https://core.rasa.com")
print("or join community chat on https://gitter.im/RasaHQ/rasa_core")
