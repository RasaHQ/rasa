import copy
import datetime
import os
import logging
import rasa.nlu
import rasa.utils.io

from typing import Any, Optional, Text
from rasa.nlu import utils
from rasa.nlu.components.builder import ComponentBuilder
from rasa.nlu.components.pipeline import ComponentPipeline
from rasa.nlu.config.nlu import RasaNLUModelConfig
from rasa.nlu.utils.package_manager import PackageManager
from rasa.nlu.model.storage.persistor import Persistor
from rasa.nlu.model.interpreter import Interpreter
from rasa.nlu.model.metadata import Metadata
from rasa.nlu.training_data import TrainingData

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer will load the data and train all components.

    Requires a pipeline specification and configuration to use for
    the training."""

    # Officially supported languages (others might be used, but might fail)
    SUPPORTED_LANGUAGES = ["de", "en"]

    def __init__(
        self,
        cfg: RasaNLUModelConfig,
        component_builder: Optional[ComponentBuilder] = None,
        skip_validation: bool = False,
    ):

        self.config = cfg
        self.skip_validation = skip_validation
        self.training_data = None  # type: Optional[TrainingData]

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in
            # a new builder. hence, no components are reused.
            component_builder = ComponentBuilder()

        # Before instantiating the component classes, lets check if all
        # required packages are available
        if not self.skip_validation:
            PackageManager.validate_requirements(cfg.component_names)

        # build pipeline
        self.pipeline = self._build_pipeline(cfg, component_builder)
        print("cool")

    @staticmethod
    def _build_pipeline(
        cfg: RasaNLUModelConfig, component_builder: ComponentBuilder
    ) -> ComponentPipeline:
        """Transform the passed names of the pipeline components into classes"""
        pipeline = ComponentPipeline()

        # Transform the passed names of the pipeline components into classes
        for i in range(len(cfg.pipeline)):
            component_cfg = cfg.for_component(i)
            component = component_builder.create_component(component_cfg, cfg)
            pipeline.add_component(component)

        return pipeline

    def train(self, data: TrainingData, **kwargs: Any) -> "Interpreter":
        """Trains the underlying pipeline using the provided training data."""

        self.training_data = data

        self.training_data.validate()

        context = kwargs

        for component in self.pipeline:
            updates = component.provide_context()
            if updates:
                context.update(updates)

        # Before the training starts: check that all arguments are provided
        if not self.skip_validation:
            self.pipeline.validate(context)

        # data gets modified internally during the training - hence the copy
        working_data = copy.deepcopy(data)

        for i, component in enumerate(self.pipeline):
            logger.info("Starting to train component {}".format(component.name))
            component.prepare_partial_processing(self.pipeline.get_component(i), context)
            updates = component.train(working_data, self.config, **context)
            logger.info("Finished training component.")
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context)

    @staticmethod
    def _file_name(index, name):
        return "component_{}_{}".format(index, name)

    def persist(
        self,
        path: Text,
        persistor: Optional[Persistor] = None,
        fixed_model_name: Text = None,
    ) -> Text:
        """Persist all components of the pipeline to the passed path.

        Returns the directory of the persisted model."""

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        metadata = {"language": self.config["language"], "pipeline": []}

        if fixed_model_name:
            model_name = fixed_model_name
        else:
            model_name = "nlu_" + timestamp

        path = os.path.abspath(path)
        dir_name = os.path.join(path, model_name)

        rasa.utils.io.create_directory(dir_name)

        if self.training_data:
            metadata.update(self.training_data.persist(dir_name))

        for i, component in enumerate(self.pipeline):
            file_name = self._file_name(i, component.name)
            update = component.persist(file_name, dir_name)
            component_meta = component.component_config
            if update:
                component_meta.update(update)
            component_meta["class"] = utils.module_path_from_object(component)

            metadata["pipeline"].append(component_meta)

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.persist(dir_name, model_name)
        logger.info(
            "Successfully saved model into '{}'".format(os.path.abspath(dir_name))
        )
        return dir_name
