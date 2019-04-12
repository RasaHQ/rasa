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
	rasa shell -m models --endpoints endpoints.yml

run-core:
	rasa run core -m models --endpoints endpoints.yml

run-actions:
	python -m rasa_core_sdk.endpoint --actions actions

train-core:
	rasa train core -d domain.yml -s data/stories.md -o models -c config.yml

train-interactive:
	rasa interactive core -d domain.yml -m models -c config.yml --data data

visualize:
	rasa show stories
