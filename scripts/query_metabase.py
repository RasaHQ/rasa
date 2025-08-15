import requests
import json
import os

METABASE_URL = "https://rasa.metabaseapp.com"
METABASE_API_KEY = os.environ.get("METABASE_API_KEY", "")
VERSION = os.environ.get("RASA_PRO_VERSION", "")

if not METABASE_API_KEY:
    raise Exception("METABASE_API_KEY environment variable not set")

query = f"""
SELECT
  "rasa_open_source"."identifies"."context_rasa_pro" AS "rasa_pro_version",
  CASE WHEN "rasa_open_source"."identifies"."context_docker" THEN 'Docker' ELSE 'Python' END AS "artefact",
  COUNT(*) AS "count"
FROM
  "rasa_open_source"."identifies"
WHERE
  "rasa_open_source"."identifies"."context_rasa_pro" = '{VERSION}'
GROUP BY
  1, 2
ORDER BY
  (string_to_array("context_rasa_pro", '.'))[1]::integer DESC,
  (string_to_array("context_rasa_pro", '.'))[2]::integer DESC,
  substring((string_to_array("context_rasa_pro", '.'))[3] FROM E'^\\d+')::integer DESC,
  substring((string_to_array("context_rasa_pro", '.'))[3] FROM E'\\d+$')::integer DESC,
  2 ASC
"""

headers = {"X-Api-Key": METABASE_API_KEY, "Content-Type": "application/json"}

query_response = requests.post(
    f"{METABASE_URL}/api/dataset",
    headers=headers,
    json={
        "native": {"query": query},
        "type": "native",
        "database": 2,
    }
)

def get_artefact_counts():
    query_response.raise_for_status()
    results = query_response.json()
    artefact_counts = {row[1]: row[2] for row in results["data"]["rows"]}
    docker = artefact_counts.get("Docker", 0)
    python = artefact_counts.get("Python", 0)
    return docker, python

if __name__ == "__main__":
    docker, python = get_artefact_counts()
    print(f"DOCKER={docker}")
    print(f"PYTHON={python}")

