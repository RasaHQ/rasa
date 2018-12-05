from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open("rasa_core/version.py").read())

# Get the long description from the README file
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

tests_requires = [
    "pytest~=3.0",
    "pytest-pep8~=1.0",
    "pytest-cov~=2.0",
    "pytest_localserver~=0.4.0",
    "treq~=17.0",
    "freezegun~=0.3.0",
    "nbsphinx>=0.3",
    "matplotlib~=2.0",
    "responses~=0.9.0",
    "httpretty~=0.9.0",
]

install_requires = [
    "jsonpickle~=0.9.0",
    "redis~=2.0",
    "fakeredis~=0.10.0",
    "pymongo~=3.5",
    "future~=0.16",
    "numpy~=1.14",
    "scipy~=1.1",
    "typing~=3.0",
    "requests~=2.20",
    "tensorflow==1.10.0",
    "h5py~=2.0",
    "apscheduler~=3.0",
    "tqdm~=4.0",
    "ConfigArgParse~=0.13.0",
    "networkx~=2.0",
    "fbmessenger~=5.0",
    "pykwalify<=1.6.0",
    "coloredlogs~=10.0",
    "ruamel.yaml~=0.15.0",
    "flask~=1.0",
    "flask_cors~=3.0",
    "scikit-learn~=0.19.0",
    "slackclient~=1.0",
    "python-telegram-bot~=10.0",
    "twilio~=6.0",
    "webexteamssdk~=1.0",
    "mattermostwrapper~=2.0",
    "rocketchat_API~=0.6.0",
    "colorhash~=1.0",
    "pika~=0.11.2",
    "jsonschema~=2.6",
    "packaging~=17.0",
    "gevent~=1.2",
    "pytz~=2018.4",
    "python-dateutil~=2.7",
    "rasa_nlu~=0.13.0",
    "rasa_core_sdk~=0.12.1",
    "colorclass~=2.2",
    "terminaltables~=3.1",
    "PyInquirer~=1.0",
    "prompt_toolkit==1.0.14",
    "flask-jwt-simple~=0.0.3",
    "python-socketio>=2.1.1,<3",
    "pydot~=1.2",
]

extras_requires = {
    "test": tests_requires
}

setup(
    name="rasa-core",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        # supported python versions
        "Programming Language :: Python",
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
    author="Rasa Technologies GmbH",
    author_email="hi@rasa.com",
    maintainer="Tom Bocklisch",
    maintainer_email="tom@rasa.com",
    license="Apache 2.0",
    keywords="nlp machine-learning machine-learning-library bot bots "
             "botkit rasa conversational-agents conversational-ai chatbot"
             "chatbot-framework bot-framework",
    url="https://rasa.com",
    download_url="https://github.com/RasaHQ/rasa_core/archive/{}.tar.gz"
                 "".format(__version__),
    project_urls={
        "Bug Reports": "https://github.com/rasahq/rasa_core/issues",
        "Source": "https://github.com/rasahq/rasa_core",
    },
)

print("\nWelcome to Rasa Core!")
print("If any questions please visit documentation "
      "page https://rasa.com/docs/core")
print("or join the community discussions on https://forum.rasa.com")
