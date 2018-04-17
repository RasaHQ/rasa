
Roadmap
=======


These are features we are keen on! Check out issues in github to see status of things.

- easier deployment options
- create a platform for rasa NLU users to share models/ data
- support loading training data from a DB instead of a text file
- entity normalisation: as is, the named entity extractor will happily extract `cheap` & `inexpensive` as entities of the `expense` class, but will not tell you that these are realisations of the same underlying concept. You can easily handle that with a list of aliases in your code, but we want to offer a more elegant & generalisable solution. [Word Forms](https://github.com/gutfeeling/word_forms) looks promising.
- support for more (human) languages
