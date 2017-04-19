FROM python:2.7-slim

ENV RASA_NLU_DOCKER="YES" \
    RASA_NLU_HOME=/app \
    RASA_NLU_PYTHON_PACKAGES=/usr/local/lib/python2.7/dist-packages

VOLUME ["${RASA_NLU_HOME}", "${RASA_NLU_PYTHON_PACKAGES}"]

# Run updates, install basics and cleanup
# - build-essential: Compile specific dependencies
# - git-core: Checkout git repos
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
  build-essential \
  git-core && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./requirements.txt ${RASA_NLU_HOME}/requirements.txt

RUN pip install -r "${RASA_NLU_HOME}/requirements.txt"

COPY . ${RASA_NLU_HOME}

WORKDIR ${RASA_NLU_HOME}

RUN echo "-e .\n" | cat - ./requirements.txt > /tmp/out && mv /tmp/out ./requirements.txt

RUN python setup.py install

RUN ls /app

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["help"]
