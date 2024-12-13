import socketio


class RasaSocketIOClient:
    def __init__(self, url="http://localhost:5005"):
        self.sio = socketio.AsyncClient()
        self.setup_handlers()
        self.bot_responses = []
        self.session_id = None
        self.url = url

    def setup_handlers(self):
        @self.sio.event
        async def connect():
            print("Connected!")
            await self.sio.emit("session_request", {"session_id": None})

        @self.sio.event
        def disconnect():
            print("Disconnected!")

        @self.sio.on("bot_uttered")
        def on_bot_message(data):
            print("Received bot message:", data)
            if isinstance(data, dict) and "text" in data:
                print(f"Bot: {data['text']}")
                self.bot_responses.append(data["text"])

        @self.sio.on("session_confirm")
        async def on_session_confirm(data):
            if isinstance(data, str):
                self.session_id = data
                print(f"Session ID: {self.session_id}")
            elif isinstance(data, dict) and "session_id" in data:
                self.session_id = data["session_id"]
                print(f"Session ID: {self.session_id}")
            else:
                print("Invalid session_confirm data:", data)

    async def connect_to_server(self):
        print(f"Connecting to server at {self.url}")
        await self.sio.connect(self.url)

    async def send_message(self, message: str):
        print(f"Sending message: {message}")
        await self.sio.emit(
            "user_uttered", {"message": message, "session_id": self.session_id}
        )
