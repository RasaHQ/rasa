.PHONY: clean test lint

TEST_PATH=./

help:
	@echo "    train-nlu"
	@echo "        Train the natural language understanding using Rasa NLU."
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."
	@echo "    run"
	@echo "        Spin up a server that serves as an endpoint to receive facebook user messages."

train-nlu:
	python -m rasa_nlu.train -c nlu_model_config.json --fixed_model_name current

train-core:
	python -m rasa_core.train -s data/stories.md -d domain.yml -o models/dialogue --epochs 300

run:
	python -m rasa_core.run -d models/dialogue -u models/nlu/current -p 5002 -c facebook --credentials credentials.yml

run-cmdline:
	python -m rasa_core.run -d models/dialogue -u models/nlu/current
