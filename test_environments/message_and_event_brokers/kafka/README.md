# Testing Kafka broker

This is a set of various configurations under which Kafka message broker can be run.
Use them to spin up Kafka as a Docker container in a predefined configuration.

Configurations include:
* [Kafka without any authentication](no_authentication/README.md)
* [Kafka with SASL_PLAIN authentication without TLS encryption](sasl_plain/no_tls/README.md)
* [Kafka with SASL_PLAIN authentication and TLS encryption](sasl_plain/with_tls/README.md)
* [Kafka with SASL_SCRAM authentication without TLS encryption](sasl_scram/no_tls/README.md)
* [Kafka with SASL_SCRAM authentication and TLS encryption](sasl_scram/with_tls/README.md)
