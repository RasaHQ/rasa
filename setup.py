from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('rasa_core/version.py').read())

tests_requires = [
    "pytest-pep8",
    "pytest-services",
    "pytest-flask",
    "pytest-cov",
    "pytest-xdist"
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
    'graphviz',
    'Keras',
    'tensorflow',
    'h5py',
    'apscheduler',
    'tqdm',
    'ConfigArgParse',
    'networkx',
    'pymessenger',
]

extras_requires = {
    'test': tests_requires
}

setup(
    name='rasa_core',
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6"
    ],
    packages=find_packages(exclude=["tests", "tools"]),
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    description="Rasa dialogue manager",
    author='Rasa Technologies GmbH',
    author_email='hi@rasa.ai',
    url='https://github.com/RasaHQ/rasa_core'
)
