from __future__ import annotations

from typing import Dict, Iterator, Text, Union, List, Tuple
from pathlib import Path
import numpy as np
import csv

from rasa.core.evaluation.marker_base import EventMetaData


def compute_statistics(
    values: List[Union[float, int]]
) -> Dict[Text, Union[int, np.float]]:
    """Computes some statistics over the given numbers."""
    return {
        "count": len(values) if values else 0,
        "mean": np.mean(values) if values else np.nan,
        "median": np.median(values) if values else np.nan,
        "min": min(values) if values else np.nan,
        "max": max(values) if values else np.nan,
    }


class MarkerStatistics:
    """Computes some statistics on marker extraction results.

    (1) Per dialogue:

    For each marker, we consider all relevant events where that marker applies.
    Everytime a marker applies, we check how many user turns precede that event.
    We collect all these numbers and compute basic statistics (e.g. count and mean)
    on them.

    This means, per dialogue, we compute how often a marker applies and how many
    user turns preceding a relevant marker application on average, in that dialogue.

    (2) Over all dialogues:

    Here, for each marker, we consider all relevant events where a marker applies
    *in any of the dialogues*. Then, we again calculate basic statistics over the
    respective number of user turns that precedes each of these events.

    This means, over all dialogues, we compute to how many events the marker applies
    in total and how many user turns preceding a relevant marker application,
    on average, if applies to some event in a dialogue.

    Moreover we compute:
    - the total number of dialogues and
    - the total number (and percentage) of dialogues in which each marker applies at
      least once.
    """

    NO_MARKER = "-"
    STAT_NUM_DIALOGUES = "total_number_of_dialogues"
    STAT_NUM_DIALOGUES_WHERE_APPLIES = (
        "number_of_dialogues_where_marker_applies_at_least_once"
    )
    STAT_PERCENTAGE_DIALOGUES_WHERE_APPLIES = (
        "percentage_of_dialogues_where_marker_applies_at_least_once"
    )

    @staticmethod
    def _full_stat_name(stat_name: Text) -> Text:
        return f"{stat_name}(number of preceding user turns)"

    ALL_DIALOGUES = np.nan
    ALL_SENDERS = "all"

    def __init__(self) -> None:
        """Creates a new marker statistic."""
        # to ensure consistency of processed rows
        self._marker_names = []

        # (1) For the per-dialogue analysis of the "preceeding user turns":
        # For each marker and each possible statistic, we collect a list of the
        # respective value for each time that the marker applies.
        self.dialogue_results: Dict[Text, Dict[Text, List[Union[np.float, int]]]] = {}
        # To keep track of where the marker applies, we remember the indices in a
        # separate list:
        self.dialogue_results_indices: List[Tuple[Text, int]] = []

        # (2) For the overall statistics on the "preceeding user turns":
        # Since the median cannot be computed easily from a stream, we have to
        # collect all variables.
        self.num_preceeding_user_turns_collected: Dict[Text, List[int]] = {}

        # (3) For the remaining overall statistics:
        self.count_if_applied_at_least_once: Dict[Text, int] = {}
        self.num_dialogues = 0

    def process(
        self,
        extracted_markers: Dict[Text, List[EventMetaData]],
        dialogue_idx: int,
        sender_id: Text,
    ) -> None:
        """Adds the restriction result to the statistics evaluation.

        Args:
            extracted_markers: marker extraction results, i.e. a dictionary mapping
                from a marker name to a list of meta data describing relevant events
                for that marker
            dialogue_idx: an index that, together with the `sender_id` identifies
                the dialogue from which the markers where extracted
            sender_id: an id that, together with the `dialogue_idx` identifies
                the dialogue from which the markers where extracted
        """
        if len(self._marker_names) == 0:
            # sort and initialise here once so our result tables are sorted
            self._marker_names = sorted(extracted_markers.keys())
            self.count_if_applied_at_least_once = {
                marker_name: 0 for marker_name in self._marker_names
            }
            self.num_preceeding_user_turns_collected = {
                marker_name: [] for marker_name in self._marker_names
            }
            self.dialogue_results = {
                marker_name: {} for marker_name in self._marker_names
            }
        else:
            if set(extracted_markers.keys()) != set(self.dialogue_results):
                raise RuntimeError(
                    f"Expected all processed extraction results to contain information"
                    f"for the same set of markers. But found "
                    f"{set(extracted_markers.keys())} which differs from "
                    f"the marker extracted so far (i.e. {self.dialogue_results})."
                )

        self.num_dialogues += 1
        # update row-idx of per tracker statistics
        self.dialogue_results_indices.append((sender_id, dialogue_idx))

        for marker_name, meta_data in extracted_markers.items():

            num_preceeding_user_turns = [
                event_meta_data.preceding_user_turns for event_meta_data in meta_data
            ]

            # update columns of per tracker statistics
            marker_results = self.dialogue_results[marker_name]
            statistics = compute_statistics(num_preceeding_user_turns)
            for stat_name, stat_value in statistics.items():
                full_stat_name = self._full_stat_name(stat_name)
                marker_results.setdefault(full_stat_name, []).append(stat_value)

            # update overall statistics
            self.num_preceeding_user_turns_collected[marker_name].extend(
                num_preceeding_user_turns
            )
            if len(num_preceeding_user_turns):
                self.count_if_applied_at_least_once[marker_name] += 1

    def to_csv(self, path: Path, overwrite: bool = False) -> None:
        """Exports the resulting statistics to a csv file.

        Args:
            path: path to where the csv file should be written.
            overwrite: set to `True` to enable overwriting an existing file
        """
        if path.is_file() and not overwrite:
            raise FileExistsError(f"Expected that there was no file at {path}.")
        with path.open(mode="w") as f:
            table_writer = csv.writer(f)

            # columns
            table_writer.writerow(self._header())

            # write overall statistics first
            special_sender_idx = self.ALL_SENDERS
            special_dialogue_idx = self.ALL_DIALOGUES
            row = self._as_row(
                sender_id=special_sender_idx,
                dialogue_idx=special_dialogue_idx,
                marker_name=self.NO_MARKER,
                statistic_name=self.STAT_NUM_DIALOGUES,
                value=self.num_dialogues,
            )
            table_writer.writerow(row)

            for marker_name, count in self.count_if_applied_at_least_once.items():
                row = self._as_row(
                    sender_id=special_sender_idx,
                    dialogue_idx=special_dialogue_idx,
                    marker_name=marker_name,
                    statistic_name=self.STAT_NUM_DIALOGUES_WHERE_APPLIES,
                    value=count,
                )
                table_writer.writerow(row)
                row = self._as_row(
                    sender_id=special_sender_idx,
                    dialogue_idx=special_dialogue_idx,
                    marker_name=marker_name,
                    statistic_name=self.STAT_PERCENTAGE_DIALOGUES_WHERE_APPLIES,
                    value=(count / self.num_dialogues * 100)
                    if self.num_dialogues
                    else 100.0,
                )
                table_writer.writerow(row)

            # write the per-dialogue statistics
            for marker_name, statistics in self.dialogue_results.items():
                for row in self._as_rows(
                    marker_name=marker_name,
                    statistics=statistics,
                    row_indices=self.dialogue_results_indices,
                ):
                    table_writer.writerow(row)

    @staticmethod
    def _header() -> List[Text]:
        return [
            "sender_id",
            "dialogue_idx",
            "marker",
            "statistic",
            "value",
        ]

    @staticmethod
    def _as_row(
        sender_id: Text,
        dialogue_idx: Union[int, np.float],
        marker_name: Text,
        statistic_name: Text,
        value: Union[int, np.float],
    ) -> List[Text]:
        if isinstance(value, int):
            value_str = str(value)
        elif np.isnan(value):
            value_str = str(np.nan)
        else:
            value_str = np.round(value, 3)
        return [
            str(item)
            for item in [
                sender_id,
                dialogue_idx,
                marker_name,
                statistic_name,
                value_str,
            ]
        ]

    @staticmethod
    def _as_rows(
        marker_name: Text,
        statistics: Dict[Text, List[Union[int, np.float]]],
        row_indices: List[Tuple[Text, int]],
    ) -> Iterator[List[str]]:
        for statistic_name in sorted(statistics):
            values = statistics[statistic_name]
            for (sender_id, dialogue_idx), value in zip(row_indices, values):
                yield MarkerStatistics._as_row(
                    sender_id=sender_id,
                    dialogue_idx=dialogue_idx,
                    marker_name=marker_name,
                    statistic_name=statistic_name,
                    value=value,
                )
