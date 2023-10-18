from sanic import Sanic, response
from sanic.request import Request
from sanic.response import HTTPResponse


def create_app() -> Sanic:

    bot_app = Sanic("callback_server", configure_logging=False)

    @bot_app.post("/bot")
    def print_response(request: Request) -> HTTPResponse:
        """Print bot response to the console."""
        bot_response = request.json.get("text")
        print(f"\n{bot_response}")

        body = {"status": "message sent"}
        return response.json(body, status=200)

    return bot_app


if __name__ == "__main__":
    app = create_app()
    port = 5034

    print(f"Starting callback server on port {port}.")
    app.run("0.0.0.0", port)
