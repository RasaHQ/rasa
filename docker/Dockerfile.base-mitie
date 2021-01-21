# The base image used for all images that require a MITIE model
FROM alpine:latest

RUN apk add --update make wget

# download mitie model
WORKDIR /build
COPY ./Makefile .
RUN make prepare-mitie
