FROM python:2.7-slim

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
  libpng12-dev \
  curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
  mkdir /app

WORKDIR /app

# Copy as early as possible so we can cache ...
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . /app

RUN pip install -e .

VOLUME ["/app/model"]

EXPOSE 5005

ENTRYPOINT ["./entrypoint.sh"]

CMD ["start", "--core", "./model/dialogue", "--nlu", "./model/nlu"]
