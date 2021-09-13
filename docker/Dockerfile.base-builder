# The base builder image used for all images
ARG IMAGE_BASE_NAME
ARG POETRY_VERSION

FROM ${IMAGE_BASE_NAME}:base-poetry-${POETRY_VERSION}

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
  libffi7 \
  libffi-dev \
  libpng-dev \
  && apt-get autoremove -y

# Make sure that all security updates are installed
RUN apt-get update && apt-get dist-upgrade -y --no-install-recommends
