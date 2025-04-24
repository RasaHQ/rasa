# Testing message and event brokers

This directory contains configurations and instructions on how to set up and run various message and event brokers.

## Supported Services
Supported services include:
* [kafka](kafka/README.md)

## Generating Certificates
Certificates are required for secure communication between clients and brokers.
Kafka can use certificates for authentication and encryption if they are provided through keystore.

Makefile provides a convenient way to generate certificates and import them into the keystore.
Run the following command:
```bash
make generate-certs-and-import-into-keystore
```

This will generate a keystore with a self-signed certificate and import it into the keystore.

All generated certificates are stored in the `certs` directory.

### Configuring Generated Certificates
The generated certificates are valid for 365 days and can be used for testing purposes.
Targets provided in the makefile can be modified to generate certificates with different parameters.

These parameters include:
```text
# Common Name (CN) in the certificate' subject
COMMON_NAME = localhost

# Organisational Unit (OU) in the certificate' subject
ORGANISATION_UNIT = Atom

# Organisation (O) in the certificate's subject
ORGANIZATION = Rasa

# Location (L) in the certificate's subject
LOCATION = Berlin

# State (S) in the certificate's subject
STATE = Germany

# Country (C) in the certificate's subject
COUNTRY = GE

# SAN (Subject Alternate Name) in the certificate's subject
# It provides a mechanism to specify additional host names and IP addresses
# that the certificate should be valid for.
# Can be in format: IP:<ip address> or DNS:<dns name of the server>,
# or DNS:<dns name of the server>,IP:<ip address>,DNS:<dns name of the server> etc
SAN = DNS:localhost,IP:0.0.0.0,DNS:kafka-broker

# Password for the CA private key
CERT_PASSWORD = 123456

# Password for the keystore
CERT_LIFETIME_IN_DAYS = 365

# Kafka's keystore file name
SERVER_KEY_STORE = server.keystore.jks
# Password for the keystore
KEYSTORE_PASSWORD = 123456
```

## Troubleshooting Certificates

#### Inspect content of the keystore
Use the following command:
```shell
make inspect-key-store
```

Parameters:
* SERVER_KEY_STORE - name of the keystore file
* KEYSTORE_PASSWORD - password for the keystore

#### Check if private key is password protected
Use the following command:
```shell
make check-private-key-password
```

Parameters:
* CERT_PASSWORD - password for the certificate's private key

#### Check if CA certificate can unlock signed certificate
Use the following command:
```shell
make ca-cert-can-unlock-signed-cert
```

### Check if TLS connection is working against running Kafka broker
Use the following command:
```shell
make check-tls-connection
```