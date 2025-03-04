# Observability
## `Open Telemetry` tracing and metrics
If using the provider endpoint and config files in this directory, to manually train and test a Rasa bot assistant, then traces and metrics from this exercise can be generated and sent to a monitoring backend for exploration and visualisation as follows:

1. To export traces and metrics to `Honeycomb` cloud, set API key `HONEYCOMB_API_KEY` (available in `1Password`): `export HONEYCOMB_API_KEY=<MY_HONEYCOMB_API_KEY>`.
    - If intending to send to a different monitoring backend, then update [`otel-collector-config.yml`](./otel-collector-config.yml) and [`otel-docker-compose.yml`](./otel-docker-compose.yml) files accordingly, and set required API key(s) if required.
2. To enable mapping traces or metrics to corresponding `rasa-pro` version, `rasa-private` git repository branch and commit hash, and `rasa-calm-demo` bot branch; set `OTEL_RESOURCE_ATTRIBUTES`
via `` `make set-otel-resource-attributes` `` (note the backticks in the command. Those are to be included, and _not_ to be omitted).
    - If intending to set other attributes also, such as git tag etc, then set `OTEL_RESOURCE_ATTRIBUTES` accordingly
    manually: `export OTEL_RESOURCE_ATTRIBUTES=key1=value,key2=value2,key-n=value-n`
3. Start OTEL collector as a `docker` container (named `otel-collector`): `make run-otel-collector`
4. Train and use Rasa bot assistant.
5. View traces and metrics: Now generated traces and metrics would be visible in:
    - `Honeycomb`'s [web UI](https://ui.honeycomb.io/rasa/environments/engine) (or other chosen monitoring backend, if using a different one).
    - Also in `otel-collector`'s console logs. To troubleshoot in case traces or metrics do not appear on the monitoring backend, start by checking these logs via: `make print-otel-collector-logs`
