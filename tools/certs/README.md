# TLS Certificate Generation

This directory contains tooling to generate a CA, client, and server TLS certificates using [`cfssl`](https://github.com/cloudflare/cfssl).

## Prerequisites

Install `cfssl` for your platform:

**macOS**
```bash
make install-cfssl-mac-os
```

**Ubuntu**
```bash
make install-cfssl-ubuntu
```

## Generating Certificates

### Generate all certificates at once

```bash
make generate-certs
```

This will generate the CA, client, and server certificates in the current directory.

### Generate individual certificates

| Command | Description |
|---|---|
| `make generate-ca` | Generate the Certificate Authority (CA) |
| `make generate-client-cert` | Generate the client certificate (also generates CA) |
| `make generate-server-cert` | Generate the server certificate (also generates CA) |

## Customising the Server Certificate Hostname

The `HOSTNAME` environment variable controls the list of hostnames and IP addresses for which the server certificate is valid. 

It defaults to:

```
localhost,127.0.0.1,action-server-grpc-tls,action-server-https
```

> **Important for Docker users:** When services communicate over a Docker network, containers reach each other by their DNS service name. You must include those service names in `HOSTNAME`, otherwise TLS handshakes will fail with a hostname mismatch error.

Override `HOSTNAME` when generating the server certificate:

```bash
make generate-server-cert HOSTNAME="localhost,127.0.0.1,my-service,my-other-service"
```

Or when generating all certificates at once:

```bash
make generate-certs HOSTNAME="localhost,127.0.0.1,my-service,my-other-service"
```

## Output Files

After generation, the following files will be present in this directory:

| File | Description |
|---|---|
| `ca.pem` | CA certificate (public) |
| `ca-key.pem` | CA private key |
| `client.pem` | Client certificate (public) |
| `client-key.pem` | Client private key |
| `server.pem` | Server certificate (public) |
| `server-key.pem` | Server private key |

