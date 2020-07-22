# This is a very simple callback server.
# Running Rasa with this channel as the
# connector will forward messages to this
# server, which will print the bot's
# responses to your console.

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/bot", methods=["POST"])
def print_response():
    bot_response = request.json
    print(bot_response.get("text"))

    return {"status": "message sent"}


if __name__ == "__main__":
    app.run()
