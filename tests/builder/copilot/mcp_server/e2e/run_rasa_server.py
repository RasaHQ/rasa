#!/usr/bin/env python3
"""Script to run Rasa server for testing.

This script is used by E2E tests to start the Rasa server in a subprocess.
Usage: python run_rasa_server.py <model_path> <port> <endpoints_path>
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_rasa_server.py <model_path> <port> <endpoints_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    port = int(sys.argv[2])
    endpoints_path = Path(sys.argv[3])

    # Initialize Configuration with endpoints
    from rasa.core.config.configuration import Configuration

    Configuration.initialise_endpoints(endpoints_path).initialise_sub_agents(
        None
    ).initialise_credentials(None)

    # Run server (blocking)
    try:
        import rasa.api

        rasa.api.run(
            model=model_path,
            sub_agents="",
            connector="rest",
            port=port,
            interface="127.0.0.1",
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
