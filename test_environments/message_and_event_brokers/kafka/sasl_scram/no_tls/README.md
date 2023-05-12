# Setup of Kafka with SASL_SCRAM authentication (SHA-256 and SHA-512 algorithm) and without TLS encryption

This is a simple setup of Kafka with SASL_SCRAM authentication (SHA-256 and SHA-512 algorithm) and without TLS encryption. 
It is intended to be used to set up test environment in which Kafka brokers require clients to authenticate.
All communication is done over insecure plain connection (without TLS).
Kafka will be listening on port 9092. You can connect to the broker with URL localhost:9092.

## How to start the environment

We provide two different test environments for this setup. One with SHA-256 and one with SHA-512.

Typical startup of the environment:
1. Start Zookeeper container
2. Add two users to Zookeeper
    * One will be used by Kafka to authenticate itself to Zookeeper
    * The other will be used by the client to authenticate to Kafka
3. Start Kafka container

Details of the startup are described in the corresponding README.md files in the directories.


## How to connect to the broker

To connect to the broker from the client, use user:

| User        | Password       |
|-------------|----------------|
| kafkaclient | password       |

The user is defined in zookeeper.
You must also set the authentication method to either `SASL SCRAM-SHA-256 or SASL SCRAM-SHA-512` 
mechanism in the client's configuration. Which SCRAM SHA you will use depends on which test environment you are running.