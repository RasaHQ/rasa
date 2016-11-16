from rasa_nlu.training_data import TrainingData


def test_luis_mitie():
    td = TrainingData('data/demo-restaurants.json','mitie')
    assert td.fformat == 'luis'
    # some more assertions


def test_wit_spacy():
    td = TrainingData('data/demo-flights.json','spacy_sklearn')
    assert td.fformat == 'wit'

