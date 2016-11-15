import numpy as np

def test_spacy():
    import spacy
    from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
    nlp = spacy.load('en')
    doc=nlp(u"hey how are you today")
    ftr = SpacyFeaturizer(nlp)
    vecs = ftr.create_bow_vecs([sentence])
    assert doc.vector[:5] == np.array([-0.19649599,  0.32493639, -0.37408298, -0.10622784,  0.062756])
    assert vec[0] == doc.vector


def test_mitie():
    from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
    ftr = MITIEFeaturizer()
    sentence = "Hey how are you today"
    vecs = ftr.create_bow_vecs([sentence])
    assert vecs[0][:5] == np.array([ 0.        , -4.4551446 ,  0.26073121, -1.46632245, -1.84205751])    
