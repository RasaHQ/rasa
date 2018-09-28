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
	python3 -m rasa_core.train -s data/stories.md -d domain.yml \
				 -o models/dialogue --epochs 100 --debug

run-core:
	python3 -m rasa_core.run --core models/dialogue --debug --endpoints endpoints.yml

run-actions:
	python3 -m rasa_core_sdk.endpoint --actions actions

run:
	make run-actions&
	make run-core

visualize:
	python -m rasa_core.visualize -s data/stories.md -d domain.yml -o story_graph.png
