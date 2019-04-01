help:
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."
	@echo "    run"
	@echo "        Runs the bot on the command line."
	@echo "    run-core"
	@echo "        Runs the core server."
	@echo "    run-actions"
	@echo "        Runs the action server."
	@echo "    train-interactive"
	@echo "        Train the bot with interactive learning."
	@echo "    visualize"
	@echo "        Show your stories as a graph."

run:
	make run-actions&
	make run-core

run-core:
	python -m rasa.core.run --core models/dialogue --endpoints endpoints.yml

run-actions:
	python -m rasa_core_sdk.endpoint --actions actions

train-core:
	python train.py

train-interactive:
	make run-actions&
	python train_interactive.py

visualize:
	python visualize.py
