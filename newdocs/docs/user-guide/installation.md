# Installation

## Quick Installation

You can install Rasa Open Source using pip (requires Python 3.6 or 3.7).

```
$ pip3 install rasa
```


* Having trouble installing? Read our step-by-step installation guide.


* You can also build Rasa Open Source from source.


* For advanced installation options such as building from source and installation instructions for
custom pipelines, head over here.


* Prefer following video instructions? Watch our installation series on [Youtube](https://www.youtube.com/playlist?list=PL75e0qA87dlEWUA5ToqLLR026wIkk2evk).

When you’re done installing, you can head over to the tutorial!


---

## Step-by-step Installation Guide

### 1. Install the Python development environment

Check if your Python environment is already configured:

```
$ python3 --version
$ pip3 --version
```

If these packages are already installed, these commands should display version
numbers for each step, and you can skip to the next step.

Otherwise, proceed with the instructions below to install them.

Ubuntu

Fetch the relevant packages using `apt`, and install virtualenv using `pip`.

```
$ sudo apt update
$ sudo apt install python3-dev python3-pip
```

macOS

Install the [Homebrew](https://brew.sh) package manager if you haven’t already.

Once you’re done, you can install Python3.

```
$ brew update
$ brew install python
```

Windows

Make sure the Microsoft VC++ Compiler is installed, so python can compile
any dependencies. You can get the compiler from <a class="reference external"
href="https://visualstudio.microsoft.com/visual-cpp-build-tools/"
target="_blank">Visual Studio</a>. Download the installer and select
VC++ Build tools in the list.Install [Python 3](https://www.python.org/downloads/windows/) (64-bit version) for Windows.

```
C:\> pip3 install -U pip
```

**NOTE**: Note that pip in this refers to pip3 as Rasa Open Source requires python3. To see which version
the pip command on your machine calls use pip –version.

### 2. Create a virtual environment (strongly recommended)

Tools like [virtualenv](https://virtualenv.pypa.io/en/latest/) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) provide isolated Python environments, which are cleaner than installing packages systemwide (as they prevent dependency conflicts). They also let you install packages without root privileges.

Ubuntu / macOS

Create a new virtual environment by choosing a Python interpreter and making a `./venv` directory to hold it:

```
$ python3 -m venv ./venv
```

Activate the virtual environment:

```
$ source ./venv/bin/activate
```

Windows

Create a new virtual environment by choosing a Python interpreter and making a `.\\venv` directory to hold it:

```
C:\> python3 -m venv ./venv
```

Activate the virtual environment:

```
C:\> .\venv\Scripts\activate
```

### 3. Install Rasa Open Source

Ubuntu / macOS / Windows

First make sure your `pip` version is up to date:

```
$ pip install -U pip
```

To install Rasa Open Source:

```
$ pip install rasa
```

**Congratulations! You have successfully installed Rasa Open Source!**

You can now head over to the tutorial.


---

## Building from Source

If you want to use the development version of Rasa Open Source, you can get it from GitHub:

```
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
$ git clone https://github.com/RasaHQ/rasa.git
$ cd rasa
$ poetry install
```


---

## NLU Pipeline Dependencies

Several NLU components have additional dependencies that need to
be installed separately.

Here, you will find installation instructions for each of them below.

### How do I choose a pipeline?

The page on Choosing a Pipeline will help you pick the right pipeline
for your assistant.

### I have decided on a pipeline. How do I install the dependencies for it?

When you install Rasa Open Source, the dependencies for the `supervised_embeddings` - TensorFlow
and sklearn_crfsuite get automatically installed. However, spaCy and MITIE need to be separately installed if you want to use pipelines containing components from those libraries.

#### Dependencies for spaCy

For more information on spaCy, check out the [spaCy docs](https://spacy.io/usage/models).

You can install it with the following commands:

```
$ pip install rasa[spacy]
$ python -m spacy download en_core_web_md
$ python -m spacy link en_core_web_md en
```

This will install Rasa Open Source as well as spaCy and its language model
for the English language. We recommend using at least the
“medium” sized models (`_md`) instead of the spaCy’s
default small `en_core_web_sm` model. Small models require less
memory to run, but will somewhat reduce intent classification performance.

#### Dependencies for MITIE

First, run

```
$ pip install git+https://github.com/mit-nlp/MITIE.git
$ pip install rasa[mitie]
```

and then download the
[MITIE models](https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2).
The file you need is `total_word_feature_extractor.dat`. Save this
anywhere. If you want to use MITIE, you need to
tell it where to find this file (in this example it was saved in the
`data` folder of the project directory).

**WARNING**: Mitie support is likely to be deprecated in a future release.
