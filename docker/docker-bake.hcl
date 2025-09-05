variable "IMAGE_NAME" {
  default = "rasa/rasa"
}

variable "IMAGE_TAG" {
  default = "localdev"
}

variable "GH_OWNER" {
  default = "TAProjectGermany"
}

variable "BASE_IMAGE_HASH" {
  default = "localdev"
}

variable "BASE_MITIE_IMAGE_HASH" {
  default = "localdev"
}

variable "BASE_BUILDER_IMAGE_HASH" {
  default = "localdev"
}

# keep this in sync with the version in .github/poetry_version.txt
# the variable is set automatically for builds in CI
variable "POETRY_VERSION" {
  default = "1.8.2"
}

group "base-images" {
  targets = ["base", "base-poetry", "base-mitie"]
}

target "base" {
  dockerfile = "docker/Dockerfile.base"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:base-${IMAGE_TAG}"]
  cache-to   = ["type=inline"]
}

target "base-mitie" {
  dockerfile = "docker/Dockerfile.base-mitie"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:base-mitie-${IMAGE_TAG}"]
  cache-to   = ["type=inline"]
}

target "base-poetry" {
  dockerfile = "docker/Dockerfile.base-poetry"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:base-poetry-${POETRY_VERSION}"]

  args = {
    IMAGE_BASE_NAME = "${IMAGE_NAME}"
    BASE_IMAGE_HASH = "${BASE_IMAGE_HASH}"
    POETRY_VERSION  = "${POETRY_VERSION}"
  }

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-poetry-${POETRY_VERSION}",
  ]
}

target "base-builder" {
  dockerfile = "docker/Dockerfile.base-builder"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:base-builder-${IMAGE_TAG}"]

  args = {
    IMAGE_BASE_NAME = "${IMAGE_NAME}"
    POETRY_VERSION  = "${POETRY_VERSION}"
  }

  cache-to = ["type=inline"]
}

target "default" {
  dockerfile = "Dockerfile"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:${IMAGE_TAG}"]

  args = {
    IMAGE_BASE_NAME         = "${IMAGE_NAME}"
    BASE_IMAGE_HASH         = "${BASE_IMAGE_HASH}"
    BASE_BUILDER_IMAGE_HASH = "${BASE_BUILDER_IMAGE_HASH}"
  }

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-${BASE_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-builder-${BASE_BUILDER_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:latest",
  ]
}

target "full" {
  dockerfile = "docker/Dockerfile.full"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:${IMAGE_TAG}-full"]

  args = {
    IMAGE_BASE_NAME         = "${IMAGE_NAME}"
    BASE_IMAGE_HASH         = "${BASE_IMAGE_HASH}"
    BASE_MITIE_IMAGE_HASH   = "${BASE_MITIE_IMAGE_HASH}"
    BASE_BUILDER_IMAGE_HASH = "${BASE_BUILDER_IMAGE_HASH}"
  }

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-${BASE_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-builder-${BASE_BUILDER_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:latest-full",
  ]
}

target "mitie-en" {
  dockerfile = "docker/Dockerfile.pretrained_embeddings_mitie_en"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:${IMAGE_TAG}-mitie-en"]

  args = {
    IMAGE_BASE_NAME         = "${IMAGE_NAME}"
    BASE_IMAGE_HASH         = "${BASE_IMAGE_HASH}"
    BASE_MITIE_IMAGE_HASH   = "${BASE_MITIE_IMAGE_HASH}"
    BASE_BUILDER_IMAGE_HASH = "${BASE_BUILDER_IMAGE_HASH}"
  }

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-${BASE_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-mitie-${BASE_MITIE_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-builder-${BASE_BUILDER_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:latest-mitie-en",
  ]
}

target "spacy-de" {
  dockerfile = "docker/Dockerfile.pretrained_embeddings_spacy_de"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:${IMAGE_TAG}-spacy-de"]

  args = {
    IMAGE_BASE_NAME         = "${IMAGE_NAME}"
    BASE_IMAGE_HASH         = "${BASE_IMAGE_HASH}"
    BASE_BUILDER_IMAGE_HASH = "${BASE_BUILDER_IMAGE_HASH}"
  }

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-${BASE_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-builder-${BASE_BUILDER_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:latest-spacy-de",
  ]
}

target "spacy-it" {
  dockerfile = "docker/Dockerfile.pretrained_embeddings_spacy_it"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:${IMAGE_TAG}-spacy-it"]

  args = {
    IMAGE_BASE_NAME         = "${IMAGE_NAME}"
    BASE_IMAGE_HASH         = "${BASE_IMAGE_HASH}"
    BASE_BUILDER_IMAGE_HASH = "${BASE_BUILDER_IMAGE_HASH}"
  }

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-${BASE_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-builder-${BASE_BUILDER_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:latest-spacy-it",
  ]
}

target "spacy-en" {
  dockerfile = "docker/Dockerfile.pretrained_embeddings_spacy_en"
  tags       = ["ghcr.io/${GH_OWNER}/rasa-base:${IMAGE_TAG}-spacy-en"]

  args = {
    IMAGE_BASE_NAME         = "${IMAGE_NAME}"
    BASE_IMAGE_HASH         = "${BASE_IMAGE_HASH}"
    BASE_BUILDER_IMAGE_HASH = "${BASE_BUILDER_IMAGE_HASH}"
  }

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-${BASE_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:base-builder-${BASE_BUILDER_IMAGE_HASH}",
    "type=registry,ref=ghcr.io/${GH_OWNER}/rasa-base:latest-spacy-en",
  ]
}
