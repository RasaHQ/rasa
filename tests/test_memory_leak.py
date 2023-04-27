import abc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Text, List, Tuple, Optional, Union

import memory_profiler
import psutil
import pytest

import rasa
import rasa.shared.utils.io
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name

PROFILING_INTERVAL = 0.1

# Enable this to plot the results locally
WRITE_RESULTS_TO_DISK = False


def _custom_default_config(
    tmp_path: Union[Path, Text], epochs: int, max_history: Optional[int] = -1
) -> Text:
    # Override default config to use custom amount of epochs
    default_config = Path("rasa", "shared", "importers", "default_config.yml")
    config = rasa.shared.utils.io.read_yaml_file(default_config)

    for model_part, items in config.items():
        for item in items:
            if "epochs" in item:
                item["epochs"] = epochs
            if "max_history" in item and max_history != -1:
                item["max_history"] = None

    config_for_test = Path(tmp_path) / "test_config.yml"
    rasa.shared.utils.io.write_yaml(config, config_for_test)

    return str(config_for_test)


class MemoryLeakTest(abc.ABC):
    """Generic template for memory leak tests."""

    @property
    def max_memory_threshold_mb(self) -> float:
        return 1000

    @pytest.fixture
    @abc.abstractmethod
    def name_for_dumped_files(self) -> Text:
        raise NotImplementedError

    @abc.abstractmethod
    def function_to_profile(self) -> None:
        raise NotImplementedError

    @pytest.mark.timeout(720, func_only=True)
    def test_for_memory_leak(self, name_for_dumped_files: Text, tmp_path: Path) -> None:
        # Run as separate process to avoid other things affecting the memory usage.
        # Unfortunately `memory-profiler` doesn't work properly with
        # `multiprocessing.Process` as it can't handle the process exit
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    f"from {__name__} import {self.__class__.__name__}; "
                    f"t = {self.__class__.__name__}();"
                    f"t.function_to_profile()"
                ),
            ],
            # Force TensorFlow to use CPU so we can track the memory usage
            env={"CUDA_VISIBLE_DEVICES": "-1"},
        )

        # Wait until process is running to avoid race conditions with the memory
        # profiling
        while not psutil.pid_exists(process.pid):
            time.sleep(0.01)

        results = memory_profiler.memory_usage(
            process, interval=PROFILING_INTERVAL, include_children=True, timestamps=True
        )

        # `memory-profiler` sometimes adds `None` values at the end which we don't need
        results = [
            memory_timestamp
            for memory_timestamp in results
            if memory_timestamp is not None
        ]

        if WRITE_RESULTS_TO_DISK:
            self._write_results(name_for_dumped_files, results)

        max_memory_usage = max(results, key=lambda memory_time: memory_time[0])[0]
        assert max_memory_usage < self.max_memory_threshold_mb

    @staticmethod
    def _write_results(base_name: Text, results: List[Tuple[float, float]]) -> None:
        mprof_plot = Path(f"{base_name}_plot.txt")
        mprof_results = Path(f"{base_name}_raw.json")

        # plot this via `mprof plot mprof_result.txt`
        with open(mprof_plot, "w") as f:
            for memory, timestamp in results:
                f.write(f"MEM {memory:.6f} {timestamp:.4f}\n")

        # dump result as json to be able analyze them without re-running the test
        with open(mprof_results, "w") as f:
            f.write(json.dumps(results))


class TestNLULeakManyEpochs(MemoryLeakTest):
    """Tests for memory leaks in NLU components when training with many epochs."""

    @property
    def epochs(self) -> int:
        return 30

    @property
    def max_memory_threshold_mb(self) -> float:
        return 2200

    def function_to_profile(self) -> None:
        import rasa.model_training

        with TempDirectoryPath(get_temp_dir_name()) as temp_dir:
            rasa.model_training.train_nlu(
                _custom_default_config(temp_dir, epochs=self.epochs),
                Path("data", "test_nlu_no_responses", "sara_nlu_data.yml"),
                output=temp_dir,
            )

    @pytest.fixture()
    def name_for_dumped_files(self) -> Text:
        return (
            f"memory_usage_rasa_nlu_{rasa.__version__}_"
            f"epochs{self.epochs}_training_runs1"
        )


class TestCoreLeakManyEpochs(MemoryLeakTest):
    """Tests for memory leaks in Core policies when training with many epochs."""

    @property
    def epochs(self) -> int:
        return 200

    @property
    def max_memory_threshold_mb(self) -> float:
        return 2000

    def function_to_profile(self) -> None:
        import rasa.model_training

        with TempDirectoryPath(get_temp_dir_name()) as temp_dir:
            rasa.model_training.train_core(
                "data/test_domains/default_with_slots.yml",
                _custom_default_config(temp_dir, epochs=self.epochs, max_history=None),
                "data/test_yaml_stories/stories_defaultdomain.yml",
                output=temp_dir,
                additional_arguments={"augmentation_factor": 20},
            )

    @pytest.fixture()
    def name_for_dumped_files(self) -> Text:
        return (
            f"memory_usage_rasa_core_{rasa.__version__}_"
            f"epochs{self.epochs}_training_runs1"
        )


class TestCRFDenseFeaturesLeak(MemoryLeakTest):
    """Tests for memory leaks in NLU the CRF when using dense features."""

    @property
    def epochs(self) -> int:
        return 1

    @property
    def max_memory_threshold_mb(self) -> float:
        return 1600

    def function_to_profile(self) -> None:
        import rasa.model_training

        config = {
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyTokenizer"},
                {"name": "SpacyFeaturizer"},
                {
                    "name": "CRFEntityExtractor",
                    "features": [
                        ["pos", "pos2"],
                        [
                            "bias",
                            "prefix5",
                            "prefix2",
                            "suffix5",
                            "suffix3",
                            "suffix2",
                            "pos",
                            "pos2",
                            "digit",
                            "text_dense_features",
                        ],
                        ["pos", "pos2"],
                    ],
                },
            ]
        }

        with TempDirectoryPath(get_temp_dir_name()) as temp_dir:
            config_for_test = Path(temp_dir) / "test_config.yml"
            rasa.shared.utils.io.write_yaml(config, config_for_test)

            rasa.model_training.train_nlu(
                str(config_for_test),
                str(Path("data", "test_nlu_no_responses", "sara_nlu_data.yml")),
                output=temp_dir,
            )

    @pytest.fixture()
    def name_for_dumped_files(self) -> Text:
        return f"memory_usage_rasa_nlu_crf_dense_{rasa.__version__}_"
