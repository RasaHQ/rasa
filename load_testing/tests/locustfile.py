import uuid

from locust import HttpUser, task

RASA_BOT_URL = "http://localhost:5005"


class RasaBotBehaviour(HttpUser):

    @task
    def task1(self):
        sender_id = str(uuid.uuid4())
        self.client.post(
            f"{RASA_BOT_URL}/webhooks/rest/webhook",
            json={"sender": sender_id, "message": "Hi Rasa bot"},
        )

