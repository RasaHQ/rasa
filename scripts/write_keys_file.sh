#!/usr/bin/env python

"""Write authentication keys from environment variable to key file for packaging."""

import sys
import re
import os
import json

if __name__ == "__main__":
    keys = {
        "segment": os.environ.get("RASA_TELEMETRY_WRITE_KEY"),
        "sentry": os.environ.get("RASA_EXCEPTION_WRITE_KEY")
    }

    with open(os.path.join("rasa", "keys"), "w") as f:
        json.dump(keys, f)

    print("Dumped keys to rasa/keys")
