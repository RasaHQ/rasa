# parsa

Parsa is a library for intent classification and entity extraction. It's functionality is intended to be similar to [wit](https://wit.ai), [LUIS](https://luis.ai), or [api.ai](https://api.ai), but packaged as a library rather than a web API. 

The intended audience is mainly people developing bots. Reasons you might use this over one of the aforementioned services: 
- you don't have to hand over your data to FB/MSFT/GOOG
- you don't have to make a `https` call every time.
- you can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d).

Training your models always happens in python, whereas you can use them in two different ways: (1) by instantiating the relevant `Interpreter` subclass in your python project, or (2) by running a simple http API locally (if you're not using python). The file `src/main.py` contains an example of the latter.

## installation:
from the main dir run
`python setup.py install`

 
## Getting Started

Pretty simple really, just open your python interpreter and type:
```python
from parsa import MITIEInterpreter
interpreter = MITIEInterpreter('data/intent_classifier.dat','data/ner.dat','data/total_word_feature_extractor.dat')
interpreter.parse("hello world")  # -> {'intent':'greet','entities':[]}

# some more examples
interpreter.parse("I want some chinese food")   # -> {'intent':'inform','entities':['cuisine':'chinese']}

interpreter.parse("thanks!")  # -> {'intent':'thankyou','entities':[]}
```

Alternatively, if you want to use the http API:

```bash
cd src/
python main.py &
curl -XPOST -H "Content-type: application/json" 'http://localhost:5002/parse' -d '{"text":"hello world"}'
# 200 OK 
# {'intent':'greet','entities':[]}
```

## Training your own models
The `scripts` directory contains some (hopefully easy-to-follow) examples of how to train the models. 
Part of the intention of this library is to make it easy to train your model from existing databases/ knowledge bases. 


## NLP Backends
The current version only includes support for MITIE, but we are working on connectors for other stacks, (especially SpaCy).


## Roadmap (message me if you'd like to submit a PR)
- entity normalisation: as is, the named entity extractor will happily extract `cheap` & `inexpensive` as entities of the `expense` class, but will not tell you that these are realisations of the same underlying concept. You can easily handle that with a list of aliases in your code, but we want to offer a more elegant & generalisable solution.
- parsing structured data, e.g. dates. We might use [parsedatetime](https://pypi.python.org/pypi/parsedatetime/) or possibly wit.ai's very own [duckling](https://duckling.wit.ai/). 
- simplify the switch from existing services by:
  - adding scripts to read in LUIS and wit model exports.
  - adding optional compatibility arguments to the `/parse` endpoint to return data in LUIS/wit/api.ai format. 
- support for more languages

