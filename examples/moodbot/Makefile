help:
	@echo "    train"
	@echo "        Train a stacked model using Rasa core & NLU."
	@echo "    run-{channel}"
	@echo "        Spin up a server that uses the connector for {channel}."
	@echo "    visualize"
	@echo "        Show your stories as a graph."
	@echo "        Spin up a server that uses the connector for {channel}."
	@echo "    evaluate"
	@echo "        Test the latest stacked model located in /models."

train:
	rasa train -d domain.yml -c config.yml --out models

train-interactive:
	rasa interactive -d domain.yml -m models -c config.yml --data data

run-fb:
	rasa run -m models -p 5002 --connector facebook --credentials credentials.yml

run-slack:
	rasa run -m models -p 5002 --connector slack --credentials credentials.yml

run-telegram:
	rasa run -m models -p 5002 --connector telegram --credentials credentials.yml

run-rocketchat:
	rasa run -m models -p 5002 --connector rocketchat --credentials credentials.yml

run-mattermost:
	rasa run -m models -p 5002 --connector mattermost --credentials credentials.yml

run-twilio:
	rasa run -m models -p 5002 --connector twilio --credentials credentials.yml

run-webexteams:
	rasa run -m models -p 5002 --connector webexteams --credentials credentials.yml

run-cmdline:
	rasa shell -m models

evaluate:
	rasa test -m models

visualize:
	rasa visualize
