FROM rasa/rasa:3.6.21

WORKDIR /app
USER root
COPY --chown=1001:1001 . /app

# The gateway verifies Firebase ID tokens using Google's public certificates.
# No Firebase service-account key is needed or included in the image.
RUN pip install --no-cache-dir -r requirements-gateway.txt \
    && chown -R 1001:1001 /app
USER 1001

# Train the small Spanish starter assistant during the image build.
RUN rasa train

# Render exposes the gateway; Rasa itself binds only to 127.0.0.1:10001.
EXPOSE 10000
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1; test -n \"$RASA_AUTH_TOKEN\" || { echo 'RASA_AUTH_TOKEN is required'; exit 1; }; exec python3 /app/secure_gateway.py"]
