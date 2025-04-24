.PHONY: all

COMMON_NAME ?= localhost
ORGANISATION_UNIT ?= Atom
ORGANIZATION ?= Rasa
LOCATION ?= Berlin
STATE ?= Germany
COUNTRY ?= GE
# can be in format: IP:<ip address> or DNS:<dns name of the server>,
# or DNS:<dns name of the server>,IP:<ip address>,DNS:<dns name of the server> etc
SAN ?= DNS:localhost,IP:0.0.0.0,DNS:kafka-broker-sasl-plain-tls,DNS:kafka-broker-scram-sha-256-tls,DNS:kafka-broker-scram-sha-512-tls

CERT_PASSWORD ?= 123456

CERT_LIFETIME_IN_DAYS ?= 365

KEYSTORE_PASSWORD ?= 123456

ifndef CERT_DIRECTORY
CERT_DIRECTORY = ./
endif

CA_CERT = $(CERT_DIRECTORY)ca-cert
CA_KEY = $(CERT_DIRECTORY)ca-key
SERVER_KEY_STORE ?= $(CERT_DIRECTORY)server.keystore.jks
CERT_REQUEST = $(CERT_DIRECTORY)cert-request
SIGNED_SERVER_CERT = $(CERT_DIRECTORY)signed-server-cert

create-public-and-private-key:
	openssl req -x509 -newkey rsa:4096 \
		-keyout $(CA_KEY) \
		-out $(CA_CERT) \
		-days $(CERT_LIFETIME_IN_DAYS) \
		-nodes \
		-subj '/CN=$(COMMON_NAME)/OU=$(ORGANISATION_UNIT)/O=$(ORGANIZATION)/L=$(LOCATION)/ST=$(STATE)/C=$(COUNTRY)' \
		-addext 'subjectAltName=$(SAN)' \
		-passin pass:$(CERT_PASSWORD) \
		-passout pass:$(CERT_PASSWORD)

create-server-key-store:
	keytool -dname "CN=localhost,OU=Atom,O=Rasa,L=Berlin,S=Germany,C=GE" \
		-keystore $(SERVER_KEY_STORE) \
		-alias localhost \
		-validity $(CERT_LIFETIME_IN_DAYS) \
		-genkey \
		-keyalg RSA \
		-storetype pkcs12 \
		-ext SAN=$(SAN) \
		-storepass $(KEYSTORE_PASSWORD) \
		-keypass $(KEYSTORE_PASSWORD)


create-certificate-request:
	keytool -keystore $(SERVER_KEY_STORE) \
		-alias localhost \
		-certreq \
		-file $(CERT_REQUEST) \
		-storepass 123456 \
		-keypass 123456 \
		-ext "SAN=$(SAN)"

create-sign-server-certificate:
	openssl x509 -req \
		-CA $(CA_CERT) \
		-CAkey $(CA_KEY) \
		-in $(CERT_REQUEST) \
		-out $(SIGNED_SERVER_CERT) \
		-days $(CERT_LIFETIME_IN_DAYS) \
		-passin pass:$(CERT_PASSWORD)


import-root-certificate-into-keystore:
	keytool -keystore $(SERVER_KEY_STORE) \
		-alias CARoot \
		-import \
		-file $(CA_CERT) \
		-storepass $(KEYSTORE_PASSWORD) \
		-keypass $(KEYSTORE_PASSWORD)

import-signed-certificate-into-keystore:
	keytool -noprompt \
		-keystore $(SERVER_KEY_STORE) \
		-alias localhost \
		-import \
		-file $(SIGNED_SERVER_CERT) \
		-storepass $(KEYSTORE_PASSWORD) \
		-keypass $(KEYSTORE_PASSWORD) \
		-ext "SAN=$(SAN)"


generate-certs-and-import-into-keystore: create-public-and-private-key create-server-key-store create-certificate-request create-sign-server-certificate import-root-certificate-into-keystore import-signed-certificate-into-keystore

inspect-key-store:
	keytool -list -v -keystore $(SERVER_KEY_STORE) -storepass $(KEYSTORE_PASSWORD) -keypass $(KEYSTORE_PASSWORD)

check-private-key-password:
	openssl rsa -check -in $(CA_KEY) -passin pass:$(CERT_PASSWORD)

ca-cert-can-unlock-signed-cert:
	openssl verify -CAfile $(CA_CERT) $(SIGNED_SERVER_CERT)

# TLS type
# TLS_TYPE can be tls1, tls1_1, tls1_2
# default is tls1 (TLS 1.0)
TLS_TYPE ?= tls1

KAFKA_BROKER_URL = localhost:29092
check-tls-connection:
	openssl s_client -debug -connect $(KAFKA_BROKER_URL) -$(TLS_TYPE)
