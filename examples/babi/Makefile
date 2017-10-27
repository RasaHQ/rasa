
run:
	python -m rasa_core.run -d "examples/babi/models/policy/current" -u 'examples/babi/models/nlu/current'

train-nlu:
	python -m rasa_nlu.train -c data/config_nlu.json --fixed_model_name current
