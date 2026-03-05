-- Create the second DB required by the deletion-no-env-expiry-true Rasa container.
-- Runs during Postgres first-time init (docker-entrypoint-initdb.d).
CREATE DATABASE deletion_no_env_expiry_true_db;
