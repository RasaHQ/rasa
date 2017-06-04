from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import rasa_nlu.version

__version__ = rasa_nlu.version.__version__

if __name__ == '__main__':
    import spacy
    nlp = spacy.load("en")
    t = nlp("My website is at http://my-webpage.com. please visit!")
    print(t)