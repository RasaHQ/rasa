help:
	@echo "    run"
	@echo "        Runs the bot on the command line."
	@echo "    run-actions"
	@echo "        Runs the action server."
	@echo "    run-core"
	@echo "        Runs the core server."
	@echo "    train-nlu"
	@echo "        Train the natural language understanding using Rasa NLU."
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."
	@echo "    train"
	@echo "        Train a stacked model using Rasa core & NLU."

run:
	rasa shell -m models --endpoints endpoints.yml

run-actions:
	python -m rasa_core_sdk.endpoint --actions actions

run-core:
	rasa run -m models --endpoints endpoints.yml

train-nlu:
	rasa train nlu -u data/nlu.md

train-core:
	rasa train core -s data/stories.md

train:
	rasa train -d data
