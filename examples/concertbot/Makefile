help:
	@echo "    run"
	@echo "        Runs the bot on the command line."
	@echo "    run-core"
	@echo "        Runs the core server."
	@echo "    run-actions"
	@echo "        Runs the action server."
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."
	@echo "    train-interactive"
	@echo "        Train the bot with interactive learning."
	@echo "    visualize"
	@echo "        Show your stories as a graph."

run:
	make run-actions&
	make run-core

run-core:
	rasa run -m models --endpoints endpoints.yml

run-actions:
	rasa run actions

train-core:
	rasa train core -d domain.yml -s data/stories.md -o models -c config.yml

train-interactive:
	rasa interactive -d domain.yml -m models -c config.yml --data data

visualize:
	rasa show stories
