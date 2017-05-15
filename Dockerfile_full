FROM rasa/rasa_nlu:latest

RUN apt-get update -qq && apt-get install -y --no-install-recommends wget \
    && wget -P data/ https://s3-eu-west-1.amazonaws.com/mitie/total_word_feature_extractor.dat \
    && apt-get remove -y wget

COPY dev-requirements.txt .
RUN pip install -r dev-requirements.txt

RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-1.2.0/en_core_web_sm-1.2.0.tar.gz --no-cache-dir > /dev/null \
    && python -m spacy link en_core_web_sm en \
    && pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_md-1.0.0/de_core_news_md-1.0.0.tar.gz --no-cache-dir > /dev/null \
    && python -m spacy link de_core_news_md de
