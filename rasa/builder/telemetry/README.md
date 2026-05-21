# `rasa/builder/telemetry/`

Two independent observability integrations for the builder.

- **Langfuse** — LLM observability: traces, spans, token usage, costs.
  Per-request, hierarchical. Used to inspect what a single copilot turn did.
- **Segment** — product analytics: discrete events forwarded to the data
  warehouse (and from there, Metabase). Used to aggregate usage across
  users, projects, and time.

Either can be enabled or disabled independently of the other.

## Directory map

```
langfuse_integration/
├── langfuse_compat.py                  # single import boundary for `langfuse`
├── shared.py                           # generation-span / session-id helpers
├── <component>_langfuse_telemetry.py   # one per traced component
└── traced_mcp_server.py                # MCP tool calls as in-trace spans

segment_integration/
├── segment_compat.py                   # single transport boundary (track())
├── shared.py                           # event-name constants + helpers
├── copilot_segment_telemetry.py        # copilot user/bot turn events
└── mcp_tools_segment_telemetry.py      # MCP tool called / error events
```

Adding instrumentation for a new component: copy the closest neighbour in
the relevant `*_integration/` package and register the call at the
component's entry point. Both packages enforce a single backend-import
boundary — never import `langfuse` or call `send_segment_request` outside
the `*_compat` modules.

## Enabling locally

### Langfuse

Requires the optional `monitoring` extra:

```
pip install 'rasa-pro[monitoring]'
```

Set:

```
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
```

If `langfuse` isn't installed or the keys are missing, every tracing call
becomes a no-op.

### Segment

Set:

```
COPILOT_SEGMENT_WRITE_KEY=...
```

Events are sent only when both `COPILOT_SEGMENT_WRITE_KEY` is present and
the global `rasa.telemetry.is_telemetry_enabled()` returns True. With
neither, every `track()` call is a no-op and logs
`builder.telemetry.track.disabled` at debug level. With the key set, a
startup log line `builder.telemetry.enabled` confirms activation.
