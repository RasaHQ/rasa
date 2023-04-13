This is a simple setup of Kafka with authentication and without TLS encryption. 
It is intended to be used to set up test environment in which Kafka brokers require clients to authenticate.
All communication is done over insecure plain connection (without TLS).
Kafka will be listening on port 9092. You can connect to the broker with URL localhost:9092.

To run:
```shell
docker-compose up -d
```

To connect to the broker from the client, use one of the users:

| User   | Password     |
|--------|--------------|
| admin  | admin-secret |
| alice  | alice-secret |

These users are defined in the `kafka_jaas.conf` file.