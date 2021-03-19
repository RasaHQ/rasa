import abc
from pathlib import Path
from typing import Text, List, Tuple, Callable, Any

import pytest

import rasa
import memory_profiler


def _config_for_epochs(tmp_path: Path, epochs: int) -> Text:
    import rasa.shared.utils.io

    # Override default config to use custom amount of epochs
    default_config = Path("rasa", "shared", "importers", "default_config.yml")
    config = rasa.shared.utils.io.read_yaml_file(default_config)
    for model_part, items in config.items():
        for item in items:
            if "epochs" in item:
                item["epochs"] = epochs
    config_for_test = tmp_path / "test_config.yml"
    rasa.shared.utils.io.write_yaml(config, config_for_test)

    return str(config_for_test)


class MemoryLeakTest(abc.ABC):
    """Generic template for memory leak tests."""

    @pytest.fixture
    @abc.abstractmethod
    def function_to_profile(self) -> Callable[[], Any]:
        raise NotImplementedError

    @pytest.fixture
    @abc.abstractmethod
    def name_for_dumped_files(self) -> Text:
        raise NotImplementedError

    def wramp_up_time_seconds(self) -> int:
        return 15

    def cooldown_time_seconds(self) -> int:
        return 15

    def interval(self) -> float:
        return 0.1

    def trend_threshold(self) -> float:
        return 0.3

    def test_for_memory_leak(
        self, function_to_profile: Callable[[], Any], name_for_dumped_files
    ) -> None:
        results = memory_profiler.memory_usage(
            function_to_profile,
            interval=self.interval(),
            timeout=3600,
            include_children=True,
            multiprocess=True,
            timestamps=True,
        )

        self._write_results(name_for_dumped_files, results)

        coefficient = self._get_coefficient_for_results(results)
        print(coefficient)
        # if this fails this indicates a memory leak!
        # Suggested Next steps:
        #   1. Run this test locally
        #   2. Plot memory usage graph which was dumped in project, e.g.:
        #      ```
        #      mprof plot \
        #        memory_usage_rasa_nlu_2.4.10_epochs1000_training_runs1_plot.txt
        #      ```
        assert coefficient < self.trend_threshold()

    def _get_coefficient_for_results(self, results: List[Tuple[float]]) -> float:
        import numpy as np
        from numpy.polynomial import polynomial

        # ignore the ramp up in the beginning and packaging at the end
        results = results[
            int(self.wramp_up_time_seconds() / self.interval()) : len(results)
            - int(self.cooldown_time_seconds() / self.interval())
        ]

        x = np.array([timestamp for (_, timestamp) in results])
        y = np.array([memory for (memory, _) in results])

        trend = polynomial.polyfit(x, y, deg=1)
        return float(trend[1])

    @staticmethod
    def _write_results(base_name: Text, results: List[Tuple[float]]) -> None:
        mprof_plot = Path(f"{base_name}_plot.txt")
        mprof_results = Path(f"{base_name}_raw.json")

        # plot this via `mprof plot mprof_result.txt`
        with open(mprof_plot, "w+") as f:
            for memory, timestamp in results:
                f.write(f"MEM {memory:.6f} {timestamp:.4f}\n")

        import json

        # dump result as json to be able analyze them without re-running the test
        with open(mprof_results, "w") as f:
            f.write(json.dumps(results))


class TestNLULeakManyEpochs(MemoryLeakTest):
    """Tests for memory leaks in NLU components when training with many epochs."""

    # [-0.00159855] for 2.4.0
    # [0.21319729] [0.18872215] for 2.2.0
    def epochs(self) -> int:
        return 1000

    @pytest.fixture()
    def function_to_profile(
        self,
        # default_nlu_data: Text,  # 2.2.0
        nlu_data_path: Text,
        tmp_path: Path,
    ) -> Callable[[], Any]:
        # from rasa.train import train_nlu  # 2.2.0

        from rasa.model_training import train_nlu

        def profiled_train() -> None:
            train_nlu(
                _config_for_epochs(tmp_path, epochs=self.epochs()),
                # default_nlu_data,  # 2.2.0
                nlu_data_path,
                output=str(tmp_path),
            )

        return profiled_train

    @pytest.fixture()
    def name_for_dumped_files(self) -> Text:
        return (
            f"memory_usage_rasa_nlu_{rasa.__version__}_"
            f"epochs{self.epochs()}_training_runs1"
        )


class TestNLULeakManyRuns(MemoryLeakTest):
    """Tests for memory leaks in NLU components when training with many epochs.

    If this fails but `TestNLULeakManyEpochs` does not, it is an indicator that
    there is a leak in the data loading pipeline.
    """

    # [-0.00159855] for 2.4.0
    # [0.21319729] [0.18872215] for 2.2.0
    def training_runs(self) -> int:
        return 30

    @pytest.fixture()
    def function_to_profile(
        self,
        # default_nlu_data: Text,  # 2.2.0
        nlu_data_path: Text,
        tmp_path: Path,
    ) -> Callable[[], Any]:
        # from rasa.train import train_nlu  # 2.2.0
        from rasa.model_training import train_nlu

        def profiled_train() -> None:
            for _ in range(self.training_runs()):
                train_nlu(
                    _config_for_epochs(tmp_path, epochs=1),
                    # default_nlu_data,  # 2.2.0
                    nlu_data_path,
                    output=str(tmp_path),
                )

        return profiled_train

    @pytest.fixture()
    def name_for_dumped_files(self) -> Text:
        return (
            f"memory_usage_rasa_nlu_{rasa.__version__}_"
            f"epochs1_training_runs{self.training_runs()}"
        )


class TestCoreLeakManyEpochs(MemoryLeakTest):
    """Tests for memory leaks in Core policies when training with many epochs."""

    # 2.4.0: [0.06253618]
    # 2.2.0: [0.35051641]
    def epochs(self) -> int:
        return 1000

    @pytest.fixture()
    def function_to_profile(
        self,
        domain_path: Text,
        stories_path: Text,
        # default_domain_path: Text,  # 2.2.0
        # default_stories_file: Text,  # 2.2.0
        tmp_path: Path,
    ) -> Callable[[], Any]:
        # from rasa.train import train_core  # 2.2.0
        from rasa.model_training import train_core

        def profiled_train() -> None:
            train_core(
                domain_path,
                # default_domain_path,  # 2.2.0
                _config_for_epochs(tmp_path, epochs=self.epochs()),
                stories_path,
                # default_stories_file,  # 2.2.0
                output=str(tmp_path),
            )

        return profiled_train

    @pytest.fixture()
    def name_for_dumped_files(self) -> Text:
        return (
            f"memory_usage_rasa_core_{rasa.__version__}_"
            f"epochs{self.epochs()}_training_runs1"
        )


class TestCoreLeakManyRuns(MemoryLeakTest):
    """Tests for memory leaks in Core policies when training with multiple runs.

    If this fails but `TestCoreLeakManyEpochs` does not, it is an indicator that
    there is a leak in the data loading pipeline.
    """

    def training_runs(self) -> int:
        return 30

    def trend_threshold(self) -> float:
        return 0.35

    @pytest.fixture()
    def function_to_profile(
        self,
        domain_path: Text,
        stories_path: Text,
        # default_domain_path: Text,  # 2.2.0
        # default_stories_file: Text,  # 2.2.0
        tmp_path: Path,
    ) -> Callable[[], Any]:
        # from rasa.train import train_core  # 2.2.0
        from rasa.model_training import train_core

        def profiled_train() -> None:
            for _ in range(self.training_runs()):
                train_core(
                    domain_path,
                    # default_domain_path,  # 2.2.0
                    _config_for_epochs(tmp_path, epochs=1),
                    stories_path,
                    # default_stories_file,  # 2.2.0
                    output=str(tmp_path),
                )

        return profiled_train

    @pytest.fixture()
    def name_for_dumped_files(self) -> Text:
        return (
            f"memory_usage_rasa_core_{rasa.__version__}_"
            f"epochs1_training_runs{self.training_runs()}"
        )
