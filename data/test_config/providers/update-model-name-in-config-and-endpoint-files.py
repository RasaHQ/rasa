#!/usr/bin/env python
"""Update LLM and Embedding models in endpoints and config files."""

import os
import sys
from enum import Enum
from glob import glob
from itertools import chain

import yaml

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")

# Azure provider keys
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT")
LLM_API_VERSION = os.getenv("LLM_API_VERSION")
EMBEDDINGS_DEPLOYMENT = os.getenv("EMBEDDINGS_DEPLOYMENT")
EMBEDDINGS_API_VERSION = os.getenv("EMBEDDINGS_API_VERSION")

# Azure and Huggingface Inference Endpoints key
LLM_API_BASE = os.getenv("LLM_API_BASE")
EMBEDDINGS_API_BASE = os.getenv("EMBEDDINGS_API_BASE")

PROVIDER_CONFIG_AND_ENDPOINTS_FILE_DIRECTORY = os.getenv("PROVIDER")
ENDPOINT_FILE_REQUIRED_COMPONENT_KEY = "nlg"
CONFIG_FILE_REQUIRED_COMPONENTS_MAP = {
    "pipeline": {
        "LLMBasedRouter",
        "MultiStepLLMCommandGenerator",
        "SingleStepLLMCommandGenerator",
    },
    "policies": {"EnterpriseSearchPolicy", "IntentlessPolicy"},
}
ENDPOINT_FILES = glob(f"{PROVIDER_CONFIG_AND_ENDPOINTS_FILE_DIRECTORY}/*endpoint*.yml")
CONFIG_FILES = glob(f"{PROVIDER_CONFIG_AND_ENDPOINTS_FILE_DIRECTORY}/*config*.yml")


class ModelConfigKey(str, Enum):
    """Config parameter keys to specify model."""

    MODEL = "model"
    DEPLOYMENT = "deployment"
    API_VERSION = "api_version"
    API_BASE = "api_base"


def new_model_group_format_config() -> bool:
    """Check new format of defining models config in `model_groups` in use."""
    if ENDPOINT_FILES:
        for endpoint_file in ENDPOINT_FILES:
            with open(endpoint_file, encoding="utf-8") as file:
                if "model_groups" in yaml.safe_load(file):
                    return True
        return False
    raise sys.exit("No endpoint file(s) found")


def update_new_model_group_format_endpoint_files() -> None:
    """Update model (or Azure deployment details) to specified one, in `model_groups`."""
    model_updated = False

    def get_model_group_by_id_containing(model_groups: list, text: str) -> list:
        return list(
            chain.from_iterable(
                [
                    model_group["models"]
                    for model_group in model_groups
                    if text in model_group["id"]
                ]
            )
        )

    for endpoint_file in ENDPOINT_FILES:
        with open(endpoint_file, encoding="utf-8") as file:
            yml_data = yaml.safe_load(file)
            model_groups = yml_data["model_groups"]
            llm_models_group = get_model_group_by_id_containing(model_groups, "llm")
            embeddings_models_group = get_model_group_by_id_containing(
                model_groups, "embedding"
            )
            for llm_model in llm_models_group:
                if LLM_MODEL_NAME and ModelConfigKey.MODEL in llm_model:
                    llm_model[ModelConfigKey.MODEL] = LLM_MODEL_NAME
                    model_updated = True
                if LLM_DEPLOYMENT and ModelConfigKey.DEPLOYMENT in llm_model:
                    llm_model[ModelConfigKey.DEPLOYMENT] = LLM_DEPLOYMENT
                    model_updated = True
                if LLM_API_VERSION and ModelConfigKey.API_VERSION in llm_model:
                    llm_model[ModelConfigKey.API_VERSION] = LLM_API_VERSION
                    model_updated = True
                if LLM_API_BASE and ModelConfigKey.API_BASE in llm_model:
                    llm_model[ModelConfigKey.API_BASE] = LLM_API_BASE
                    model_updated = True
            for embedding_model in embeddings_models_group:
                if EMBEDDINGS_MODEL_NAME and ModelConfigKey.MODEL in embedding_model:
                    embedding_model[ModelConfigKey.MODEL] = EMBEDDINGS_MODEL_NAME
                    model_updated = True
                if (
                    EMBEDDINGS_DEPLOYMENT
                    and ModelConfigKey.DEPLOYMENT in embedding_model
                ):
                    embedding_model[ModelConfigKey.DEPLOYMENT] = EMBEDDINGS_DEPLOYMENT
                    model_updated = True
                if (
                    EMBEDDINGS_API_VERSION
                    and ModelConfigKey.API_VERSION in embedding_model
                ):
                    embedding_model[ModelConfigKey.API_VERSION] = EMBEDDINGS_API_VERSION
                    model_updated = True
                if EMBEDDINGS_API_BASE and ModelConfigKey.API_BASE in embedding_model:
                    embedding_model[ModelConfigKey.API_BASE] = EMBEDDINGS_API_BASE
                    model_updated = True
        if model_updated:
            with open(endpoint_file, "w", encoding="utf-8") as file:
                yaml.dump(yml_data, stream=file)


def update_old_format_endpoint_files() -> None:
    """Update model (or Azure deployment details) to specified one, in endpoints.yml."""
    model_updated = False

    for endpoint_file in ENDPOINT_FILES:
        with open(endpoint_file, encoding="utf-8") as file:
            yml_data = yaml.safe_load(file)
            if (nlg := yml_data.get(ENDPOINT_FILE_REQUIRED_COMPONENT_KEY)) and (
                nlg_llm := nlg.get("llm")
            ):
                if LLM_MODEL_NAME and ModelConfigKey.MODEL in nlg_llm:
                    nlg_llm[ModelConfigKey.MODEL] = LLM_MODEL_NAME
                    model_updated = True
                if LLM_DEPLOYMENT and ModelConfigKey.DEPLOYMENT in nlg_llm:
                    nlg_llm[ModelConfigKey.DEPLOYMENT] = LLM_DEPLOYMENT
                    model_updated = True
                if LLM_API_VERSION and ModelConfigKey.API_VERSION in nlg_llm:
                    nlg_llm[ModelConfigKey.API_VERSION] = LLM_API_VERSION
                    model_updated = True
                if LLM_API_BASE and ModelConfigKey.API_BASE in nlg_llm:
                    nlg_llm[ModelConfigKey.API_BASE] = LLM_API_BASE
                    model_updated = True
        if model_updated:
            with open(endpoint_file, "w", encoding="utf-8") as file:
                yaml.dump(yml_data, stream=file)


def update_config_files() -> None:
    """Update model (or Azure deployment details) to specified one, in config files."""
    model_updated = False

    def update_embeddings_model_config(component_block: dict) -> None:
        if not (embeddings := component_block.get("embeddings")):
            return
        nonlocal model_updated
        if EMBEDDINGS_MODEL_NAME and ModelConfigKey.MODEL in embeddings:
            embeddings[ModelConfigKey.MODEL] = EMBEDDINGS_MODEL_NAME
            model_updated = True
        if EMBEDDINGS_DEPLOYMENT and ModelConfigKey.DEPLOYMENT in embeddings:
            embeddings[ModelConfigKey.DEPLOYMENT] = EMBEDDINGS_DEPLOYMENT
            model_updated = True
        if EMBEDDINGS_API_VERSION and ModelConfigKey.API_VERSION in embeddings:
            embeddings[ModelConfigKey.API_VERSION] = EMBEDDINGS_API_VERSION
            model_updated = True
        if EMBEDDINGS_API_BASE and ModelConfigKey.API_BASE in embeddings:
            embeddings[ModelConfigKey.API_BASE] = EMBEDDINGS_API_BASE
            model_updated = True

    for config_file in CONFIG_FILES:
        with open(config_file, encoding="utf-8") as file:
            yml_data = yaml.safe_load(file)
            for (
                component,
                sub_components,
            ) in CONFIG_FILE_REQUIRED_COMPONENTS_MAP.items():
                for sub_component in sub_components:
                    if not (
                        ("singlestep" in config_file and "Multi" in sub_component)
                        or ("multistep" in config_file and "Single" in sub_component)
                        or ("dut" in config_file and sub_component == "LLMBasedRouter")
                    ):
                        if (
                            sub_component_block := next(
                                (
                                    item
                                    for item in yml_data[component]
                                    if item["name"] == sub_component
                                ),
                                None,
                            )
                        ) and (llm := sub_component_block.get("llm")):
                            if LLM_MODEL_NAME and ModelConfigKey.MODEL in llm:
                                llm[ModelConfigKey.MODEL] = LLM_MODEL_NAME
                                model_updated = True
                            if LLM_DEPLOYMENT and ModelConfigKey.DEPLOYMENT in llm:
                                llm[ModelConfigKey.DEPLOYMENT] = LLM_DEPLOYMENT
                                model_updated = True
                            if LLM_API_VERSION and ModelConfigKey.API_VERSION in llm:
                                llm[ModelConfigKey.API_VERSION] = LLM_API_VERSION
                                model_updated = True
                            if LLM_API_BASE and ModelConfigKey.API_BASE in llm:
                                llm[ModelConfigKey.API_BASE] = LLM_API_BASE
                                model_updated = True
                        if sub_component_block:
                            update_embeddings_model_config(sub_component_block)
                            if "flow_retrieval" in sub_component_block:
                                update_embeddings_model_config(
                                    sub_component_block["flow_retrieval"]
                                )
        if model_updated:
            with open(config_file, "w", encoding="utf-8") as file:
                yaml.dump(yml_data, stream=file)


if __name__ == "__main__" and (
    LLM_MODEL_NAME
    or EMBEDDINGS_MODEL_NAME
    or LLM_DEPLOYMENT
    or LLM_API_VERSION
    or LLM_API_BASE
    or EMBEDDINGS_DEPLOYMENT
    or EMBEDDINGS_API_VERSION
    or EMBEDDINGS_API_BASE
):
    if not PROVIDER_CONFIG_AND_ENDPOINTS_FILE_DIRECTORY:
        raise sys.exit("Directory for endpoint and config files not provided")
    if new_model_group_format_config():
        update_new_model_group_format_endpoint_files()
    else:
        update_old_format_endpoint_files()
        update_config_files()
