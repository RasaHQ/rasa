#!/usr/bin/env bash
# Bootstrap RDS with assistant database and user.
# Connect as the RDS master user (e.g. integration_tests_user).
#
# Required env: PGHOST, PGUSER, PGPASSWORD,
#   DB_ASSISTANT_USERNAME, DB_ASSISTANT_DATABASE
# Optional: PGPORT (default 5432), PGDATABASE (default postgres)

set -euo pipefail

for var in PGHOST PGUSER PGPASSWORD DB_ASSISTANT_USERNAME DB_ASSISTANT_DATABASE; do
  if [ -z "${!var:-}" ]; then
    echo "Missing required env: $var" >&2
    exit 1
  fi
done

# PGHOST must be hostname only (no :port); strip :port if present (e.g. from db_host:port)
if [[ "${PGHOST}" == *:* ]]; then
  PGPORT="${PGHOST##*:}"
  PGHOST="${PGHOST%%:*}"
fi
export PGHOST
export PGPORT="${PGPORT:-5432}"
export PGDATABASE="${PGDATABASE:-postgres}"
export PGSSLMODE="${PGSSLMODE:-require}"

# Quote identifiers for SQL: double-quote and escape any " inside (prevents SQL injection)
pg_quote_ident() { echo "\"$(echo "$1" | sed 's/"/""/g')\""; }
u_quoted=$(pg_quote_ident "$DB_ASSISTANT_USERNAME")
d_quoted=$(pg_quote_ident "$DB_ASSISTANT_DATABASE")

echo "create assistant database and user"
echo "CREATE USER $u_quoted;" | psql -qtAX
echo "GRANT rds_iam TO $u_quoted;" | psql -qtAX
echo "CREATE DATABASE $d_quoted WITH ENCODING = 'UTF8';" | psql -qtAX
echo "GRANT ALL PRIVILEGES ON DATABASE $d_quoted TO $u_quoted;" | psql -qtAX
echo "ALTER DATABASE $d_quoted OWNER TO $u_quoted;" | psql -qtAX

echo "create user integration_tests_user_no_password (no password)"
echo "CREATE USER integration_tests_user_no_password;" | psql -qtAX
echo "GRANT rds_iam TO integration_tests_user_no_password;" | psql -qtAX
echo "GRANT ALL PRIVILEGES ON DATABASE integrationtestsdb TO integration_tests_user_no_password;" | psql -qtAX
PGDATABASE=integrationtestsdb psql -qtAX -c "GRANT CREATE ON SCHEMA public TO integration_tests_user_no_password;"

# Update the RDS IAM auth user with all required permissions here.