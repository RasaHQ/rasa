FROM python:3.6-slim as builder
# if this installation process changes, the enterprise container needs to be
# updated as well
WORKDIR /build
COPY . .
RUN python setup.py sdist bdist_wheel
RUN find dist -maxdepth 1 -mindepth 1 -name '*.tar.gz' -print0 | xargs -0 -I {} mv {} rasa.tar.gz

FROM python:3.6-slim

SHELL ["/bin/bash", "-c"]

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
  libpq-dev \
  curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
  mkdir /install

WORKDIR /install

# Copy as early as possible so we can cache ...
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY --from=builder /build/rasa.tar.gz .
RUN pip install ./rasa.tar.gz[sql]

VOLUME ["/app"]
WORKDIR /app

EXPOSE 5005

ENTRYPOINT ["rasa"]

CMD ["--help"]
