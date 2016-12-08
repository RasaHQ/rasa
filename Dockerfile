FROM python:2.7-slim
MAINTAINER p@dialoganalytics.com

ENV PORT 5000

# Run updates, install basics and cleanup
# - build-essential: Compile specific dependencies
# - git-core: Checkout git repos
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
  build-essential \
  git-core && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up app directory
RUN mkdir -p /app
WORKDIR /app

# Install dependencies, use cache if possible
COPY . /app
RUN pip install -r requirements.txt
RUN python setup.py install

# Download spacy data
RUN python -m spacy.en.download all
RUN python -m spacy.de.download all

EXPOSE 5000

ENTRYPOINT ["python", "-m rasa_nlu.server"]
