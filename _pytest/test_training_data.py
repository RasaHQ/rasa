from rasa_nlu.training_data import TrainingData


def test_luis_mitie():
    td = TrainingData('data/demo-restaurants.json','mitie')
    assert td.fformat == 'luis'
    # some more assertions


def test_wit():
    assert True
