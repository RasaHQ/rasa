# parsa

Parsa is a library for intent classification and entity extraction. It's functionality is intended to be similar to [wit](https://wit.ai), [LUIS](https://luis.ai), or [api.ai](https://api.ai), but packaged as a library rather than a web API. 

The intended audience is mainly people developing bots. Reasons you might use this over one of the aforementioned services: 
- you don't have to hand over your data to FB/MSFT/GOOG
- you don't have to make a `https` call every time.
- you can tune models to work well on your particular use case.

These points are laid out in more detail in a [blog post](https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d).

Training your models always happens in python, whereas you can use them in two different ways: (1) by instantiating the relevant `Interpreter` subclass in your python project, or (2) by running a simple http API locally (if you're not using python). The file `src/main.py` contains an example of the latter.

 
## Getting Started

Pretty simple really, just open your python interpreter and type:
```
from parsa import MITIEInterpreter
interpreter = MITIEInterpreter('data/intent_classifier.dat','data/ner.dat','data/total_word_feature_extractor.dat')
interpreter.parse("hello world")
# -> {'intent':'greet','entities':[]}
interpreter.parse("I want some chinese food")
# -> {'intent':'inform','entities':['cuisine':'chinese']}
interpreter.parse("thanks!")
# -> {'intent':'thankyou','entities':[]}
```

Alternatively, if you want to use the http API:

```
cd src/
python main.py &
curl -XPOST -H "Content-type: application/json" 'http://localhost:5000/parse' -d '{"text":"hello world"}'
# 200 OK 
# {'intent':'greet','entities':[]}
```

## Training your own models
