help:
	@echo "    run"
	@echo "        Runs the bot on the command line."
	@echo "    run-actions"
	@echo "        Runs the action server."
	@echo "    run-core"
	@echo "        Runs the core server."
	@echo "    train"
	@echo "        Train a stacked model using Rasa core & NLU."

run:
	make run-actions&
	make run-core

run-actions:
	python -m rasa_core_sdk.endpoint --actions actions

run-core:
	rasa shell -m models --endpoints endpoints.yml

train:
	rasa train
