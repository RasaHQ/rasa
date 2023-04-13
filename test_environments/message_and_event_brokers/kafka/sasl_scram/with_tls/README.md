# Setup Kafka broker with SASL SCRAM authentication and TLS encryption

This is a simple setup of Kafka with SASL_SCRAM authentication and TLS encryption. 
It is intended to be used to set up test environment in which Kafka brokers require clients to authenticate.
All communication is done over secure TLS connection.

These test environments provide set up for Kafka instance which requires clients to authenticate 
over secure TLS connection.

Test environments are location in directories:
* `scram_sha_256` - Sets up Kafka with SCRAM SHA-256 algorithm enabled. Broker certificate has SAN set to `localhost`.
* `scram_sha_512` - Sets up Kafka with SCRAM SHA-512 algorithm enabled. Broker certificate has SAN set to `localhost`.

All certificates required for TLS are already generated. **They are not intended to be used in production.**
<br>Certificates in this environment, are used to verify the identity of the Kafka broker to the clients.
Pre-generated certificates are valid through `30/3/2024`. 
If you need to generate new certificates checkout the 
[How to generate certificates for TLS bound to DNS localhost](#how-to-generate-certificates-for-tls-bound-to-dns-localhost) 
section.

#### About Subject Alternate Name (SAN)

A Subject Alternate Name (or SAN) certificate is a digital security 
certificate which allows multiple hostnames or IPs to be protected by a single certificate.

Examples:
<br>If TLS certificate has SAN set to `194.3.5.1`, then clients will accept the certificate if IP of Kafka broker
to which they are connecting is `194.3.5.1`.
<br>If TLS certificate has SAN set to `localhost`, then clients will accept the certificate if the hostname
of the Kafka broker to which we are connecting is `localhost`.
<br>If TLS certificate has SAN set to `rasa.com`, then clients will accept the certificate if the hostname
of the Kafka broker to which we are connecting is `rasa.com`.
<br>If TLS certificate has SAN set to 0.0.0.0  then clients will accept the certificate 
from any hostname or IP address. This is useful for testing purposes. DO NOT USE THIS IN PRODUCTION!!!


## How to connect to Kafka broker

To connect to the broker from the client use:
* URL localhost:9098 (SASL SCRAM SHA 256) or localhost:9099 (SASL SCRAM SHA 512)
* one of the users

    | User        | Password     |
    |-------------|--------------|
    | kafkaclient | password     |

You must also enable TLS and set the authentication method to either `SASL SCRAM-SHA-256 or SASL SCRAM-SHA-512` 
mechanism in the client's configuration. Which SCRAM SHA you will use depends on which test environment you are running.

Add CA certificate to the client's certificate pool before starting it.
<br>This is required to verify the identity of the Kafka broker.
<br>If you want to skip verification of the Kafka broker's identity, 
you can instruct your client to skip verification of the certificate.
If you skip verification of the certificate communication over secure TLS will still be used, 
but the identity of the Kafka broker will not be verified.

## About certificates

RSA algorithm is used to generate private and public keys used to sign/verify/encrypt/decrypt data.
<br>
Certificate Authority (CA) is a trusted entity which issues (signs) certificates from other entities.
Certificate Authority generates a private key and a public key.
<br>Private key is used to sign certificates from other entities. It must be protected and not shared.
<br>Public key is used to verify the signature of the certificate. It is shared with the clients.
Clients use the public key to verify that the certificate 
it received was signed by the CA and not by a malicious entity.

In TLS communication we need to have:
* a CA's private key
* a CA's public key, also called `CA certificate`
* a certificate of the Kafka broker signed by CA's private key

When Kafka broker is contacted by a client, 
it sends its certificate to the client.
<br>Client verifies the certificate using the CA certificate. 
If the certificate is valid, the client can connect to the broker.

More about TLS can be read here: https://www.cloudflare.com/learning/ssl/transport-layer-security-tls/


### How to generate certificates for TLS bound to DNS localhost

This section explains how to produce certificates for the Kafka brokers and store them in the server's keystore.
We create a certificate authority (CA), also known as root certificate, 
and use it to sign the certificate request for the Kafka broker.
Kafka broker will send its signed certificate to the client. 
Client will use the CA certificate to verify the identity of the Kafka broker.

You can create the certificates and store them in the keystore using the following commands:
```shell
# Create private and public key (public key is usually reffered to as Certificate Authority's certificate or CA certificate)
openssl req -x509 -newkey rsa:4096 -keyout ca-key -out ca-cert -days 365 -nodes -subj '/CN=localhost/OU=Atom/O=Rasa/L=Berlin/ST=Germany/C=GE' -passin pass:123456 -passout pass:123456

# Create server keystore protected with storepass and keypass
keytool -dname "CN=localhost,OU=Atom,O=Rasa,L=Berlin,S=Germany,C=GE" -keystore server.keystore.jks -alias localhost -validity 365 -genkey -keyalg RSA -storetype pkcs12 -ext SAN=IP:localhost -storepass 123456 -keypass 123456

# Create a certificate request
keytool -keystore server.keystore.jks -alias localhost -certreq -file cert-request -storepass 123456 -keypass 123456 -ext "SAN=IP:localhost"

# Sign the certificate request
openssl x509 -req -CA ca-cert -CAkey ca-key -in cert-request -out signed-server-cert -days 365 -CAcreateserial -passin pass:123456

# Import root certificate into keystore
keytool -keystore server.keystore.jks -alias CARoot -import -file ca-cert -storepass 123456 -keypass 123456
# Import signed certificate into keystore
keytool -noprompt -keystore server.keystore.jks -alias localhost -import -file signed-server-cert -storepass 123456 -keypass 123456 -ext "SAN=DNS:localhost"
```


## Troubleshooting

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