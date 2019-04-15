help:
	@echo "    train"
	@echo "        Train a stacked model using Rasa core and NLU."
	@echo "    run"
	@echo "        Spin up both core and the action server"
	@echo "    visualize"
	@echo "        Show your stories as a graph"


train:
	rasa train --debug

run:
	rasa shell core -m models --verbose --endpoints endpoints.yml

train-interactive:
	rasa interactive core -d domain.yml -m models -c config.yml --data data --verbose --endpoints endpoints.yml

visualize:
	rasa show stories
