# Setup Kafka broker with SASL SCRAM (SHA-256) authentication and TLS encryption

This is a simple setup of Kafka with SASL_SCRAM (SHA-256 algorithm) authentication and TLS encryption.

To run:
```shell
# Start Zookeeper
docker-compose up -d zookeeper

# Create kafkabroker and kafkaclient users on zookeeper
docker exec -it zookeeper-scram-sha-256-tls bash
cd /etc/kafka/client
KAFKA_OPTS="-Djava.security.auth.login.config=zookeeper_client_jaas.conf" kafka-configs --zookeeper localhost:2188 --alter --add-config 'SCRAM-SHA-256=[iterations=4096,password=password]' --entity-type users --entity-name kafkabroker
KAFKA_OPTS="-Djava.security.auth.login.config=zookeeper_client_jaas.conf" kafka-configs --zookeeper localhost:2188 --alter --add-config 'SCRAM-SHA-256=[iterations=4096,password=password]' --entity-type users --entity-name client

# Exit from zookeeper container
exit

docker-compose up -d kafka-broker
```

To connect to the broker from the client, set the following properties in the client configuration:

* username and password to `kafkaclient` and `password` respectively
* SASL mechanism to `SCRAM-SHA-256`
* (optional) enable TLS encryption (different clients have different ways of doing this) or skip verification of the certificate
* import the certificate from `ca-cert` from `ssl` directory to the client's certificate pool.

