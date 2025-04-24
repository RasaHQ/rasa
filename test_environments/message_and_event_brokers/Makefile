.PHONY: all

CERT_DIRECTORY = ./certs/

ifndef COMMON_NAME
include certs/Makefile
endif

ifndef CURRENT_DIR
CURRENT_DIR := .
endif

KAFKA_DIR = kafka

CURRENT_KAFKA_PATH := $(CURRENT_DIR)/$(KAFKA_DIR)

KAFKA_NO_AUTH_PATH = $(CURRENT_KAFKA_PATH)/no_authentication
KAFKA_SASL_PLAIN_PATH = $(CURRENT_KAFKA_PATH)/sasl_plain
KAFKA_SASL_PLAIN_NO_TLS_PATH = $(KAFKA_SASL_PLAIN_PATH)/no_tls
KAFKA_SASL_PLAIN_WITH_TLS_PATH = $(KAFKA_SASL_PLAIN_PATH)/with_tls
KAFKA_SASL_SCRAM_PATH = $(CURRENT_KAFKA_PATH)/sasl_scram
KAFKA_SASL_SCRAM_NO_TLS_PATH = $(KAFKA_SASL_SCRAM_PATH)/no_tls
KAFKA_SASL_SCRAM_256_NO_TLS_PATH = $(KAFKA_SASL_SCRAM_NO_TLS_PATH)/scram_sha_256
KAFKA_SASL_SCRAM_512_NO_TLS_PATH = $(KAFKA_SASL_SCRAM_NO_TLS_PATH)/scram_sha_512
KAFKA_SASL_SCRAM_WITH_TLS_PATH = $(KAFKA_SASL_SCRAM_PATH)/with_tls
KAFKA_SASL_SCRAM_256_WITH_TLS_PATH = $(KAFKA_SASL_SCRAM_WITH_TLS_PATH)/scram_sha_256
KAFKA_SASL_SCRAM_512_WITH_TLS_PATH = $(KAFKA_SASL_SCRAM_WITH_TLS_PATH)/scram_sha_512


KAFKA_PATH ?=

start-kafka:
	docker-compose -f $(KAFKA_PATH)/docker-compose.yml up -d --wait

stop-kafka:
	docker-compose -f $(KAFKA_PATH)/docker-compose.yml down

start-no-auth-kafka: KAFKA_PATH := $(KAFKA_NO_AUTH_PATH)
start-no-auth-kafka: start-kafka # start kafka without authentication

stop-no-auth-kafka: KAFKA_PATH := $(KAFKA_NO_AUTH_PATH)
stop-no-auth-kafka: stop-kafka # stop kafka without authentication

start-sasl-plain-kafka: KAFKA_PATH := $(KAFKA_SASL_PLAIN_NO_TLS_PATH)
start-sasl-plain-kafka: start-kafka # start kafka with SASL_PLAIN authentication

stop-sasl-plain-kafka: KAFKA_PATH := $(KAFKA_SASL_PLAIN_NO_TLS_PATH)
stop-sasl-plain-kafka: stop-kafka # stop kafka with SASL_PLAIN authentication

start-sasl-plain-tls-kafka: KAFKA_PATH := $(KAFKA_SASL_PLAIN_WITH_TLS_PATH)
start-sasl-plain-tls-kafka: start-kafka # start kafka with SASL_PLAIN authentication and TLS

stop-sasl-plain-tls-kafka: KAFKA_PATH := $(KAFKA_SASL_PLAIN_WITH_TLS_PATH)
stop-sasl-plain-tls-kafka: stop-kafka # stop kafka with SASL_PLAIN authentication and TLS

start-sasl-scram-256-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_256_NO_TLS_PATH)
start-sasl-scram-256-kafka: start-kafka # start kafka with SASL_SCRAM_256 authentication

stop-sasl-scram-256-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_256_NO_TLS_PATH)
stop-sasl-scram-256-kafka: stop-kafka # stop kafka with SASL_SCRAM_256 authentication

start-sasl-scram-512-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_512_NO_TLS_PATH)
start-sasl-scram-512-kafka: start-kafka # start kafka with SASL_SCRAM_256 authentication

stop-sasl-scram-512-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_512_NO_TLS_PATH)
stop-sasl-scram-512-kafka: stop-kafka # stop kafka with SASL_SCRAM_256 authentication

start-sasl-scram-256-tls-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_256_WITH_TLS_PATH)
start-sasl-scram-256-tls-kafka: start-kafka # start kafka with SASL_SCRAM_256 authentication

stop-sasl-scram-256-tls-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_256_WITH_TLS_PATH)
stop-sasl-scram-256-tls-kafka: stop-kafka # stop kafka with SASL_SCRAM_256 authentication

start-sasl-scram-512-tls-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_512_WITH_TLS_PATH)
start-sasl-scram-512-tls-kafka: start-kafka # start kafka with SASL_SCRAM_256 authentication

stop-sasl-scram-512-tls-kafka: KAFKA_PATH := $(KAFKA_SASL_SCRAM_512_WITH_TLS_PATH)
stop-sasl-scram-512-tls-kafka: stop-kafka # stop kafka with SASL_SCRAM_256 authentication

