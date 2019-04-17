help:
	@echo "    train"
	@echo "        Train a stacked model using Rasa Core & NLU."
	@echo "    run"
	@echo "        Runs the bot on the command line."
	@echo "    run-actions"
	@echo "        Runs the action server."
	@echo "    run-core"
	@echo "        Runs the core server."

run:
	make run-actions&
	make run-core

run-actions:
	rasa run actions

run-core:
	rasa run -m models --endpoints endpoints.yml

train:
	rasa train
