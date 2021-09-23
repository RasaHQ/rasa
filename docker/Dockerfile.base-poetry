ARG IMAGE_BASE_NAME
ARG BASE_IMAGE_HASH

FROM ${IMAGE_BASE_NAME}:base-${BASE_IMAGE_HASH}

ARG POETRY_VERSION

# install poetry
ENV POETRY_VERSION ${POETRY_VERSION}
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH "/root/.poetry/bin:${PATH}"
