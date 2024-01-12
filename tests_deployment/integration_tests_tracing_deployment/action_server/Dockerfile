ARG IMAGE_TAG
ARG TARGET_IMAGE_REGISTRY

FROM ${TARGET_IMAGE_REGISTRY}/rasa:${IMAGE_TAG}

COPY action_server /app/action_server

WORKDIR /app

ENTRYPOINT ["python"]
CMD ["-m", "action_server"]
