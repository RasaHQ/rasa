import json
import time

import psycopg2

from tests.integration_tests.conftest import send_message_to_rasa_server


def test_sql_broker_stores_events():
    """This test verifies that the SQL broker stores events in the database."""
    # Send test message to Rasa
    sender_id, response = send_message_to_rasa_server(
        server_location="http://localhost:5005",
        message="hello",
    )

    # Connect to postgres and verify events
    conn = psycopg2.connect(
        dbname="rasa", user="rasa", password="rasa", host="localhost", port="5432"
    )
    cur = conn.cursor()

    try:
        # re-try test to give SQL broker some time to write to the table
        last_error = None
        for i in range(5):
            cur.execute(f"SELECT data FROM events WHERE sender_id = '{sender_id}'")
            events = cur.fetchall()

            try:
                assert (
                    len(events) >= 2
                )  # Should have UserUttered and BotUttered at minimum
                # Parse JSON data to get event types
                event_types = [json.loads(event[0])["event"] for event in events]
                assert "user" in event_types
                assert "bot" in event_types
                last_error = None
                break
            except AssertionError as e:
                last_error = e
                time.sleep(1)

        if last_error is not None:
            raise last_error

    finally:
        cur.close()
        conn.close()
