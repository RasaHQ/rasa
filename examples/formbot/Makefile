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
	python -m rasa.core.train -s data/stories.md -d domain.yml -o models/dialogue --debug

run-core:
	python -m rasa.core.run --core models/dialogue --nlu models/nlu/current --verbose --endpoints endpoints.yml

run-actions:
	python -m rasa.core_sdk.endpoint --actions actions

run:
	make run-actions&
	make run-core

train-interactive:
	python -m rasa.core.train interactive -s data/stories.md --nlu models/nlu/current -d domain.yml -o models/dialogue --verbose --endpoints endpoints.yml

visualize:
	python -m rasa.core.visualize -s data/stories.md -d domain.yml -o story_graph.html

train-nlu:
	python -m rasa.nlu.train -c config.yml --fixed_model_name current --data data/nlu.md -o models --project nlu --verbose
