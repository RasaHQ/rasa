FROM python:2

ADD . /rasa_nlu

WORKDIR /rasa_nlu

RUN pip install -r requirements.txt && \
    python setup.py install && \
    python -m spacy.en.download all > /dev/null && \
    python -m spacy.de.download all > /dev/null && \
    cp entrypoint.sh /sbin/entrypoint.sh && \
    chmod 777 entrypoint.sh

ENV RASA_NLU_DOCKER="YES"

EXPOSE 5000

ENTRYPOINT ["/sbin/entrypoint.sh"]
CMD ["help"]
