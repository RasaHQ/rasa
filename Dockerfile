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

WORKDIR ${RASA_NLU_HOME}

COPY ./requirements.txt requirements.txt

# Split into pre-requirements, so as to allow for Docker build caching
RUN pip install $(tail -n +2 requirements.txt)

COPY . ${RASA_NLU_HOME}

RUN python setup.py install 


#Uncomment these lines for using spacy (and comment the lines of the other models)
COPY config_spacy.json /app/config.json 
#Uncomment the language you want to use for spacy
RUN bash entrypoint.sh download spacy en
#RUN bash entrypoint.sh download spacy de

#Uncomment these lines for using mittie (and comment the lines of the other models)
#COPY config_mitie.json /app/config.json
#RUN bash entrypoint.sh download mitie

#Uncomment this line using mitie and sklearn (and comment the lines of the other models)
#COPY config_mitie_sklearn.json /app/config.json 

RUN ls /app

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["help"]
