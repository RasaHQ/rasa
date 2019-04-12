help:
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."
	@echo "    run-core"
	@echo "        Spin up the core server on the command line"
	@echo "    run-actions"
	@echo "        Spin up the action server"
	@echo "    run"
	@echo "        Spin up both core and the action server"
	@echo "    visualize"
	@echo "        Show your stories as a graph"


train-core:
	rasa train core -s data/stories.md -d domain.yml -o models --debug

run-core:
	rasa run core -m models --verbose --endpoints endpoints.yml

run-actions:
	python -m rasa_core_sdk.endpoint --actions actions

run:
	rasa shell core -m models --verbose --endpoints endpoints.yml

train-interactive:
	rasa interactive core -d domain.yml -m models -c config.yml --data data --verbose --endpoints endpoints.yml

visualize:
	rasa show stories
