FROM python:3.6-slim as python_builder
RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  openssh-client \
  graphviz-dev \
  pkg-config \
  git-core \
  openssl \
  libssl-dev \
  libffi6 \
  libffi-dev \
  libpng-dev \
  # required by psycopg2 at build and runtime
  libpq-dev \
  curl

# install poetry
# keep this in sync with the version in pyproject.toml and Dockerfile
ENV POETRY_VERSION 1.0.3
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH "/root/.poetry/bin:/opt/venv/bin:${PATH}"

# install dependencies
COPY . /opt/rasa/
RUN python -m venv /opt/venv && \
  . /opt/venv/bin/activate && \
  pip install --no-cache-dir -U pip && \
  cd /opt/rasa && \
  poetry install --no-dev --no-interaction

# start a new build stage
FROM python:3.6-slim

# copy everything from /opt
COPY --from=python_builder /opt /opt
ENV PATH="/opt/venv/bin:$PATH"

# change user
RUN chgrp -R 0 /opt/rasa && chmod -R g=u /opt/rasa
USER 1001

# Create a volume for temporary data
VOLUME /tmp

# change shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# the entry point
EXPOSE 5005
ENTRYPOINT ["rasa"]
CMD ["--help"]