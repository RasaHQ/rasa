help:
	@echo "    train-nlu"
	@echo "        Train the natural language understanding using Rasa NLU."
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."
	@echo "    run-{channel}"
	@echo "        Spin up a server that uses the connector for {channel}."
	@echo "    evaluate-nlu"
	@echo "        Evaluates the nlu data."
	@echo "    visualize"
	@echo "        Show your stories as a graph."

train-nlu:
	rasa train nlu -c config.yml --fixed_model_name current \
	       --data ./data/nlu.md --path models/

train-core:
	rasa train core -s data/stories.md -d domain.yml -o models

train-interactive:
	rasa interactive -s data/stories.md -m models -d domain.yml

run-fb:
	rasa shell core -m models -p 5002 -c facebook --credentials credentials.yml

run-slack:
	rasa shell core -m models -p 5002 -c slack --credentials credentials.yml

run-telegram:
	rasa shell core -m models -p 5002 -c telegram --credentials credentials.yml

run-rocketchat:
	rasa shell core -m models -p 5002 -c rocketchat --credentials credentials.yml

run-mattermost:
	rasa shell core -m models -p 5002 -c mattermost --credentials credentials.yml

run-twilio:
	rasa shell core -m models -p 5002 -c twilio --credentials credentials.yml

run-webexteams:
	rasa shell core -m models -p 5002 -c webexteams --credentials credentials.yml

run-cmdline:
	rasa shell core -m models

evaluate-nlu:
	rasa test nlu -m models --data ./data/nlu.md

visualize:
	rasa show
