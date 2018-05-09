from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import traceback
from typing import Dict, Union, List

from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import InvalidProjectError
from rasa_nlu.project import Project
from rasa_nlu.persistor import get_persistor
from rasa_nlu import utils

logger = logging.getLogger(__name__)


class ProjectManager(object):
    def __init__(self, project_dir, remote_storage, component_builder=None):
        # type: (str, str, ComponentBuilder) -> None
        self.project_dir = project_dir
        self.remote_storage = remote_storage

        if component_builder:
            self.component_builder = component_builder
        else:
            self.component_builder = ComponentBuilder(use_cache=True)

        self.project_store = None

    def _load_projects(self):
        self.project_store = self._create_project_store(self.project_dir)

    @classmethod
    def create(cls, project_dir, remote_storage, component_builder=None):
        # TODO: Should this class be a singleton class?
        instance = cls(project_dir, remote_storage, component_builder)
        instance._load_projects()

        return instance

    def _collect_projects(self, project_dir):
        if project_dir and os.path.isdir(project_dir):
            projects = os.listdir(project_dir)
        else:
            projects = []

        projects.extend(self._list_projects_in_cloud())
        return projects

    def _create_project_store(self, project_dir):
        # type: () -> Dict[str, Project]
        projects = self._collect_projects(project_dir)

        project_store = {}

        for project in projects:
            project_store[project] = Project(self.component_builder,
                                             project,
                                             self.project_dir,
                                             self.remote_storage)

        if not project_store:
            default_model = RasaNLUModelConfig.DEFAULT_PROJECT_NAME
            project_store[default_model] = Project(
                    project=default_model,
                    project_dir=self.project_dir,
                    remote_storage=self.remote_storage)
        return project_store

    @staticmethod
    def _list_projects(path):
        # type: (str) -> List[str]
        """List the projects in the path, ignoring hidden directories."""
        return [os.path.basename(fn)
                for fn in utils.list_subdirectories(path)]

    def _list_projects_in_cloud(self):
        # type: () -> List[str]
        try:
            p = get_persistor(self.remote_storage)
            if p is not None:
                return p.list_projects()
            else:
                return []
        except Exception as e:
            logger.exception("Failed to list projects. Make sure you have "
                             "correctly configured your cloud storage "
                             "settings. {}".format(traceback.format_exc()))
            return []

    def load_project(self, project=None):
        # type: (Union[None, str]) -> Project
        # Try to load a project from memory first, if not exists
        # try load from local file system or cloud by name,
        # if project not exists or can not be loaded correctly,
        # a InvalidProjectError exception will raise
        project = project or RasaNLUModelConfig.DEFAULT_PROJECT_NAME

        if project not in self.project_store:
            projects = self._list_projects(self.project_dir)

            cloud_provided_projects = self._list_projects_in_cloud()
            projects.extend(cloud_provided_projects)

            if project not in projects:
                raise InvalidProjectError(
                    "No project found with name '{}'.".format(project))
            else:
                self._try_load_project(project)

        return self.project_store[project]

    def _try_load_project(self, project):
        # type: (str) -> None
        # Try to load project from local storage, if failed
        # an InvalidProjectError exception will raise
        try:
            self.project_store[project] = Project(
                self.component_builder, project,
                self.project_dir, self.remote_storage)
        except Exception as e:
            raise InvalidProjectError(
                "Unable to load project '{}'. Error: {}".format(project, e))

    def pre_load(self, projects):
        logger.debug("loading %s", projects)
        for project in self.project_store:
            if project in projects:
                self.project_store[project].load_model()

    def get_projects(self):
        # type: () -> Dict[str, Project]
        # get all in-memory projects
        return self.project_store

    def project_exists(self, project):
        # type: (str) -> bool
        # check if a given project by name exists in project store memory
        return project in self.project_store

    def load_or_create_project(self, project):
        try:
            project_instance = self.load_project(project)
        except InvalidProjectError:
            # project not exists, create one
            project_instance = Project(
                self.component_builder, project,
                self.project_dir, self.remote_storage)

            # add to cache
            self.project_store[project] = project_instance

        return project_instance
