# Setup Kafka broker with SASL PLAIN authentication and TLS encryption
This is a simple setup of Kafka with authentication and TLS encryption. 
It is intended to be used to set up test environment in which Kafka brokers require clients to authenticate.
All communication is done over secure TLS connection.

These test environments provide set up for Kafka instance which requires clients to authenticate 
over secure TLS connection.

Test environments are location in directories:
* `ssl_all_connections` - Broker certificate has SAN set to `0.0.0.0`
* `ssl_localhost` - Broker certificate has SAN set to `localhost`

All certificates required for TLS are already generated. **They are not intended to be used in production.**
<br>Certificates in this environment, are used to verify the identity of the Kafka broker to the clients.
Pre-generated certificates are valid through `30/3/2024`. 
If you need to generate new certificates checkout the README 
files in directories `./ssl_all_connections` and `./ssl_localhost`.


## How to connect to Kafka broker
To connect to the broker from the client use:
* URL localhost:9092
* one of the users

    | User   | Password     |
    |--------|--------------|
    | admin  | admin-secret |
    | alice  | alice-secret |

Users which are available for clients are defined in the `kafka_jaas.conf` file.

Add CA certificate to the client certificate pool before starting it.
<br>This is required to verify the identity of the Kafka broker.
<br>If you want to skip verification of the Kafka broker's identity, 
you can instruct your client to skip verification of the certificate.
If you skip verification of the certificate communication over secure TLS will still be used, 
but the identity of the Kafka broker will not be verified.


# About certificates
Certificates consist of a CA (Certificate Authority) and a certificate signed by a certificate authority (CA).
<br>CA is used to sign the certificate of the Kafka broker. When Kafka broker is contacted by a client, 
it sends its certificate to the client.
<br>Client verifies the certificate using the CA certificate. 
If the certificate is valid, the client can connect to the broker.


# Troubleshooting
To inspect content of the keystore, you can use the following command:
```shell
keytool -list -v -keystore server.keystore.jks -storepass 123456 -keypass 123456
```

To check if private key is password protected
```shell
openssl rsa -check -in ca-key -passin pass:123456
```

To check if CA certificate can unlock signed certificate
```shell
openssl verify -CAfile ca-cert signed-server-cert
```

To check if TLS connection is working
```shell
# for TLS 1.0
openssl s_client -debug -connect localhost:29092 -tls1
# for TLS 1.1
openssl s_client -debug -connect localhost:29092 -tls1_1
# for TLS 1.2
openssl s_client -debug -connect localhost:29092 -tls1_2
```