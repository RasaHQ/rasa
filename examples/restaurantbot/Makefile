
run:
	python -m rasa_core.run -d models/policy/current -u models/nlu/current

train-nlu:
	python -m rasa_nlu.train -c nlu_model_config.yml --fixed_model_name current \
		   --data ./data/franken_data.json --path ./models/nlu
