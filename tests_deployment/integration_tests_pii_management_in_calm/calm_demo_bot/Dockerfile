ARG RASA_REPOSITORY="rasa/rasa-pro"
ARG RASA_IMAGE_TAG="3.13.0rc2"

FROM ${RASA_REPOSITORY}:${RASA_IMAGE_TAG} AS builder
USER root

# Install dependencies
ARG LATEST_RASA_PRO_VERSION="3.13.0rc2"
ENV LATEST_RASA_PRO_VERSION=$LATEST_RASA_PRO_VERSION

RUN python3 -m venv /opt/venv &&  \
    . /opt/venv/bin/activate &&  \
    pip install --no-cache-dir -U "pip==24.*" &&  \
    pip install --no-cache-dir "rasa-pro[pii]==${LATEST_RASA_PRO_VERSION}" &&  \
    # pin transformers to avoid issues with their OpenTelemetry metrics collector
    # described here: https://github.com/huggingface/transformers/issues/39143
    pip install --no-cache-dir "transformers==4.52.4"

# Download HF model
COPY ./gliner_model_download_script.py gliner_model_download_script.py
RUN python3 -m gliner_model_download_script

FROM builder
WORKDIR /app
COPY --from=builder /app/gliner_model /app/gliner_model
RUN --mount=type=bind,target=/app/gliner_model

# Update permissions
RUN chown -R 1001:0 /app/gliner_model && \
    chmod -R g=u /app/gliner_model && \
    chmod o+wr /app/gliner_model
USER 1001
RUN ls -l /app/gliner_model


