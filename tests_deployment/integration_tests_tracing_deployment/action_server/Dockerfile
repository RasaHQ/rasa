FROM rasa/rasa-plus:latest

COPY action_server /app/action_server

WORKDIR /app

ENTRYPOINT ["python"]
CMD ["-m", "action_server"]
