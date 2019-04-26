# Create common base stage
FROM python:3.6-slim as base

WORKDIR /build

# Create virtualenv to isolate builds
RUN python -m venv /build

# Make sure we use the virtualenv
ENV PATH="/build/bin:$PATH"

# Stage to build and install everything
FROM base as builder

# Copy only what we really need
COPY README.md .
COPY rasa ./rasa
COPY setup.py .

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

# Install Rasa and its dependencies
RUN pip install -e .[sql] && \
    # Remove pip from virtualenv since we don't need it anymore
    pip uninstall --yes pip

# Runtime stage which uses the virtualenv which we built in the previous stage
FROM base AS runner

# Copy virtualenv from previous stage
COPY --from=builder /build /build

WORKDIR /app

# Make sure the default group has the same permissions as the owner
RUN chmod -R g=u .

# Don't run as root
USER 1001

EXPOSE 5005

ENTRYPOINT ["rasa"]
CMD ["--help"]
