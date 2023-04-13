# Setup Kafka broker with SASL SCRAM authentication (SHA-512 algorithm) and without TLS encryption

This is a simple setup of Kafka with SASL_SCRAM authentication (SHA-512 algorithm) and without TLS encryption. 
It is intended to be used to set up test environment in which Kafka brokers require clients to authenticate.
All communication is done over insecure plain connection (without TLS).
Kafka will be listening on port 9092. You can connect to the broker with URL localhost:9092.

To run:
```shell
# Start Zookeeper
docker-compose up -d zookeeper

# Create kafkabroker and kafkaclient users on zookeeper
docker exec -it zookeeper-sasl-scram-sha-512 bash
cd /etc/kafka/client
KAFKA_OPTS="-Djava.security.auth.login.config=zookeeper_client_jaas.conf" kafka-configs --zookeeper localhost:2187 --alter --add-config 'SCRAM-SHA-512=[iterations=4096,password=password]' --entity-type users --entity-name kafkabroker
KAFKA_OPTS="-Djava.security.auth.login.config=zookeeper_client_jaas.conf" kafka-configs --zookeeper localhost:2187 --alter --add-config 'SCRAM-SHA-512=[iterations=4096,password=password]' --entity-type users --entity-name client

# Exit from zookeeper container
exit

docker-compose up -d kafka-broker
```
