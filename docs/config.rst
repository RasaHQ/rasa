.. _section_configuration:

Configuration
=============

You can provide options to rasa NLU through:

- a json-formatted config file
- environment variables
- command line arguments

Environment variables override options in your config file, 
and command line args will override any options specified elsewhere.
Environment variables are capitalised and prefixed with ``RASA_``, 
so the option ``backend`` is specified with the ``RASA_BACKEND`` env var.

Here is a list of all rasa NLU configuration options:

+--------------------------------------+-------------------------+------------------------------------------------------+
| Name: Type                           | Remarks                 | Description                                          |
+======================================+=========================+======================================================+
| ``backend: str``                     | - ``mitie``             | backend used for intent and entity                   |
|                                      | - ``spacy_sklearn``     | classification                                       |
|                                      | - ``mitie_sklearn``     |                                                      |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``config: str``                      |                         | configuration file (can only be set as               |
|                                      |                         | env var or command line option)                      |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``data: str``                        |                         | file containing training data.                       |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``emulate: str``                     | - ``wit``               | service to emulate                                   |
|                                      | - ``luis``              |                                                      |
|                                      | - ``api``               |                                                      |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``language: str``                    | - ``en`` (English)      | language of your app                                 |
|                                      | - ``de`` (German)       |                                                      |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``mitie_file: str``                  |                         | file containing ``total_word_feature_extractor.dat`` |
|                                      |                         | (see :ref:`section_backends`)                        |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``path: str``                        |                         | where trained models will be saved.                  |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``port: int``                        |                         | port on which to run server.                         |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``server_model_dirs: str or object`` |                         | dir containing the model to be used by               |
|                                      |                         | server or an object describing multiple models. see  |
|                                      |                         | :ref:`HTTP server config<section_http_config>`       |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``token: str``                       |                         | if set, all requests to server must have             |
|                                      |                         | a ``?token=<token>`` query param.                    |
|                                      |                         | see :ref:`section_auth`                              |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``response_log: str or null``        |                         | directory where logs will be saved (containing       |
|                                      |                         | queries and responses. if set to ``null`` logging    |
|                                      |                         | will be disabled                                     |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``num_threads: int``                 |                         | number of threads used during training               |
+--------------------------------------+-------------------------+------------------------------------------------------+
| ``fine_tune_spacy_ner: bool``        | only ``spacy_sklearn``  | fine tune existing spacy NER models vs               |
|                                      |                         | training from scratch                                |
+--------------------------------------+-------------------------+------------------------------------------------------+

If you want to persist your trained models to S3, there are additional configuration options,
see :ref:`section_persistence`
