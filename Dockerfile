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
ENV POETRY_VERSION 1.0.3
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH "/root/.poetry/bin:/opt/venv/bin:${PATH}"

# install dependencies
COPY README.md poetry.lock pyproject.toml setup.cfg /opt/rasa/
RUN python -m venv /opt/venv && \
  . /opt/venv/bin/activate && \
  pip install -U pip && \
  cd /opt/rasa && \
  poetry install --no-dev --no-interaction

# build and install rasa
COPY rasa /opt/rasa/rasa
RUN ls /opt/rasa
RUN ls /opt/rasa/rasa
RUN . /opt/venv/bin/activate && \
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
