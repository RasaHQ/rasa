# The default Docker image
ARG IMAGE_BASE_NAME
ARG BASE_IMAGE_HASH
ARG RASA_DEPS_IMAGE_HASH

FROM ${IMAGE_BASE_NAME}:rasa-deps-${RASA_DEPS_IMAGE_HASH} as rasa-install

ENV PATH="/opt/venv/bin:$PATH"

COPY . /build

WORKDIR /build

RUN poetry build -f wheel -n && \
  pip install --no-deps dist/*.whl && \
  rm -rf dist *.egg-info

# start a new build stage

FROM ${IMAGE_BASE_NAME}:base-${BASE_IMAGE_HASH} as runner

# copy everything from /opt/venv
COPY --from=rasa-install /opt/venv /opt/venv
RUN chmod -R o+rw /opt/venv/lib/python3.10/site-packages/pypred/**

# make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# set HOME environment variable
ENV HOME=/app

# update permissions & change user to not run as root
WORKDIR /app
RUN chgrp -R 0 /app && chmod -R g=u /app && chmod o+wr /app
USER 1001

# create a volume for temporary data
VOLUME /tmp

# change shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN ls -la /opt/venv/bin

# the entry point
EXPOSE 5005
ENTRYPOINT ["/bin/sh", "-c" , "ls -la /opt/venv/lib/python3.10/site-packages/pypred && rasa train"]
