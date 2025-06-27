This directory contains setup for external resources like message brokers and databases which
are required in order to be able to run integration tests on our CI.


## Troubleshooting

### Metrics TLS not working

The OpenTelemetry Collector is configured to use TLS.
If the rasa-pro container stops connecting to the OpenTelemetry Collector, it may be due to TLS issues.
Renew the TLS certificates by following the instructions [here](https://opentelemetry.io/docs/collector/configuration/#setting-up-certificates).
Ensure that the `csr.json` includes the OTEL collector's container name used in the `docker-compose.yml` file, for example:

```json
{
  "hosts": ["localhost", "127.0.0.1", "0.0.0.0", "otlp-collector"],
  "key": {
    "algo": "rsa",
    "size": 2048
  },
  "names": [
    {
      "O": "Metrics CI"
    }
  ]
}
```
