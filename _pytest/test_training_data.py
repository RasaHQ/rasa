from rasa_nlu.training_data import TrainingData


def test_luis_mitie():
    td = TrainingData('data/demo-restaurants.json', 'mitie', 'en')
    assert td.fformat == 'luis'
    # some more assertions


def test_wit_spacy():
    td = TrainingData('data/demo-flights.json', 'spacy_sklearn', 'en')
    assert td.fformat == 'wit'


def test_rasa_whitespace():
    td = TrainingData('data/demo-rasa.json', '', 'en')
    assert td.fformat == 'rasa_nlu'
