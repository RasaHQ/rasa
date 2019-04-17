help:
	@echo "    train"
	@echo "        Train a stacked model using Rasa Core and NLU."
	@echo "    run-core"
	@echo "        Spin up the core server on the command line"
	@echo "    run-actions"
	@echo "        Spin up the action server"
	@echo "    run"
	@echo "        Spin up both core and the action server"
	@echo "    visualize"
	@echo "        Show your stories as a graph"

run:
	make run-actions&
	make run-core

run-core:
	rasa run -m models --verbose --endpoints endpoints.yml

run-actions:
	rasa run actions

train:
	rasa train --debug

train-interactive:
	rasa interactive -d domain.yml -m models -c config.yml --data data --verbose --endpoints endpoints.yml

visualize:
	rasa show stories
