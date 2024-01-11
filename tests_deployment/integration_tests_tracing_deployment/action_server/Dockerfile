ARG IMAGE_TAG

FROM rasa/rasa:${IMAGE_TAG}

COPY action_server /app/action_server

WORKDIR /app

ENTRYPOINT ["python"]
CMD ["-m", "action_server"]
