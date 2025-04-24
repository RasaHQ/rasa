# Setup Kafka broker with SASL PLAIN authentication and TLS encryption
This is a simple setup of Kafka with authentication and TLS encryption. 
It is intended to be used to set up test environment in which Kafka brokers require clients to authenticate.
All communication is done over secure TLS connection.

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

### Certificate
Certificate required for TLS is already generated. **It should not be used in production.**
<br>Certificate in this environment is used to verify the identity of the Kafka broker to the clients.
Pre-generated certificates are valid through `16/4/2026 `.

## How to connect to Kafka broker
To connect to the broker from the client use:
* URL localhost:9094 (clients accept certificate from any IP Kafka broker my run on) or localhost:9095 (clients accept certificate only from Kafka Broker running on localhost)
* one of the users

    | User   | Password     |
    |--------|--------------|
    | admin  | admin-secret |
    | alice  | alice-secret |

Users which are available for clients are defined in the `kafka_jaas.conf` file.

Add CA certificate to the client certificate pool before starting it.
<br>
One option is to add it to the OS's certificate pool on the machine you are running the client on.
<br>
The other option is to add certificate to the client's in-memory certificate pool, which is active only during the client's runtime.
Consult documentation of the library you are using to manage certificates or library which you are 
using to connect to Kafka broker on how to add certificate to in-memory certificate pool. 
<br>This is required to verify the identity of the Kafka broker.
<br>If you want to skip verification of the Kafka broker's identity, 
you can instruct your client to skip verification of the certificate.
<br>
Consult the library you are using to connect to Kafka broker on how to skip verification of the certificate.
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

