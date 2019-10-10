# Create common base stage
FROM python:3.6-slim as base

WORKDIR /build

# Create virtualenv to isolate builds
#RUN python -m venv --copies /build

# Make sure we use the virtualenv
ENV PATH="/build/bin:$PATH"

# Stage to build and install everything
FROM base as builder

WORKDIR /src

# Install all required build libraries
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
  curl

ENV VIRTUAL_ENV=/build
RUN pip3 install virtualenv && python3 -m virtualenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only what we really need
COPY README.md .
COPY setup.py .
COPY MANIFEST.in .
COPY requirements.txt .

# Install Rasa and its dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Rasa as package
COPY rasa ./rasa
RUN pip install .[sql]

# Runtime stage which uses the virtualenv which we built in the previous stage
FROM base AS runner

# Copy virtualenv from previous stage
COPY --from=builder /build /build

WORKDIR /app

# Create a volume for temporary data
VOLUME /tmp

# Make sure the default group has the same permissions as the owner
RUN chgrp -R 0 . && chmod -R g=u .

# Don't run as root
#USER 1001

EXPOSE 5005

HEALTHCHECK --start-period=30s \
  CMD curl -f http://localhost:5005 || exit 1

ENTRYPOINT ["rasa"]
CMD ["--help"]
