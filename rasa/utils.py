import rasa_core
import rasa_nlu
import rasa


def print_versions() -> None:
    print("Rasa: {}\n"
          "Rasa Core: {}\n"
          "Rasa NLU: {}".format(rasa.__version__,
                                rasa_core.__version__,
                                rasa_nlu.__version__))
