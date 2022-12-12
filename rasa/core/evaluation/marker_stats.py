from __future__ import annotations
from typing import Dict, Text, Union, List, Tuple

from rasa.utils.io import WriteRow
from pathlib import Path
import csv

import numpy as np


from rasa.core.evaluation.marker_base import EventMetaData


def compute_statistics(
    values: List[Union[float, int]]
) -> Dict[Text, Union[int, float, np.floating]]:
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

    (1) Number of sessions where markers apply:

    For each marker, we compute the total number (as well as the percentage) of
    sessions in which a marker applies at least once.
    Moreover, we output the total number of sessions that were parsed.

    (2) Number of user turns preceding relevant events - per sessions:

    For each marker, we consider all relevant events where that marker applies.
    Everytime a marker applies, we check how many user turns precede that event.
    We collect all these numbers and compute basic statistics (e.g. count and mean)
    on them.

    This means, per session, we compute how often a marker applies and how many
    user turns precede a relevant marker application on average, in that session.

    (3) Number of user turns preceding relevant events - over all sessions:

    Here, for each marker, we consider all relevant events where a marker applies
    *in any of the sessions*. Then, we again calculate basic statistics over the
    respective number of user turns that precede each of these events.

    This means, we compute how many events the marker applies in total and we
    compute an estimate of the expected number of user turns preceding that
    precede an (relevant) event where a marker applies.
    """

    NO_MARKER = "-"
    STAT_NUM_SESSIONS = "total_number_of_sessions"
    STAT_NUM_SESSIONS_WHERE_APPLIES = (
        "number_of_sessions_where_marker_applied_at_least_once"
    )
    STAT_PERCENTAGE_SESSIONS_WHERE_APPLIES = (
        "percentage_of_sessions_where_marker_applied_at_least_once"
    )

    @staticmethod
    def _add_num_user_turns_str_to(stat_name: Text) -> Text:
        return f"{stat_name}(number of preceding user turns)"

    ALL_SESSIONS = np.nan
    ALL_SENDERS = "all"

    def __init__(self) -> None:
        """Creates a new marker statistics object."""
        # to ensure consistency of processed rows
        self._marker_names: List[Text] = []

        # (1) For collecting the per-session analysis:
        # NOTE: we could stream / compute them later instead of collecting them...
        self.session_results: Dict[Text, Dict[Text, List[Union[int, float]]]] = {}
        self.session_identifier: List[Tuple[Text, int]] = []

        # (2) For the overall statistics:
        self.num_preceding_user_turns_collected: Dict[Text, List[int]] = {}
        self.count_if_applied_at_least_once: Dict[Text, int] = {}
        self.num_sessions = 0

    def process(
        self,
        sender_id: Text,
        session_idx: int,
        meta_data_on_relevant_events_per_marker: Dict[Text, List[EventMetaData]],
    ) -> None:
        """Processes the meta data that was extracted from a single session.

        Internally, this method ..
        1. computes some statistics for the given meta data and saves it for later
        2. keeps track of the total number of sessions processed and the
           collects all metadata to be able to compute meta data over *all*

        Args:
            sender_id: an id that, together with the `session_idx` identifies
                the session from which the markers where extracted
            session_idx: an index that, together with the `sender_id` identifies
                the session from which the markers where extracted
            meta_data_on_relevant_events_per_marker: marker extraction results,
                i.e. a dictionary mapping
                marker names to the meta data describing relevant events
                for those markers
        """
        if len(self._marker_names) == 0:
            # sort and initialise here once so our result tables are sorted
            self._marker_names = sorted(meta_data_on_relevant_events_per_marker.keys())
            self.count_if_applied_at_least_once = {
                marker_name: 0 for marker_name in self._marker_names
            }
            self.num_preceding_user_turns_collected = {
                marker_name: [] for marker_name in self._marker_names
            }
            # NOTE: we could stream / compute them later instead of collecting them...
            stat_names = sorted(compute_statistics([]).keys())
            self.session_results = {
                marker_name: {stat_name: [] for stat_name in stat_names}
                for marker_name in self._marker_names
            }
        else:
            given_markers = meta_data_on_relevant_events_per_marker.keys()
            if set(given_markers) != set(self._marker_names):
                raise RuntimeError(
                    f"Expected all processed extraction results to contain information"
                    f"for the same set of markers. But found "
                    f"{sorted(given_markers)} which differs from "
                    f"the marker extracted so far (i.e. {sorted(self._marker_names)})."
                )

        # update session identifiers / count
        self.num_sessions += 1
        self.session_identifier.append((sender_id, session_idx))

        for marker_name, meta_data in meta_data_on_relevant_events_per_marker.items():

            num_preceding_user_turns = [
                event_meta_data.preceding_user_turns for event_meta_data in meta_data
            ]

            # update per session statistics
            statistics = compute_statistics(num_preceding_user_turns)
            for stat_name, stat_value in statistics.items():
                self.session_results[marker_name][stat_name].append(stat_value)

            # update overall statistics
            self.num_preceding_user_turns_collected[marker_name].extend(
                num_preceding_user_turns
            )
            if len(num_preceding_user_turns):
                self.count_if_applied_at_least_once[marker_name] += 1

    def overall_statistic_to_csv(self, path: Path, overwrite: bool = False) -> None:
        """Exports the overall statistics (over all processes sessions) to a csv file.

        Args:
            path: path to where the csv file should be written.
            overwrite: set to `True` to enable overwriting an existing file
        """
        if path.is_file() and not overwrite:
            raise FileExistsError(f"Expected that there was no file at {path}.")
        with path.open(mode="w") as f:
            table_writer = csv.writer(f)
            table_writer.writerow(self._header())
            self._write_overview(table_writer)
            self._write_overall_statistics(table_writer)

    def per_session_statistics_to_csv(
        self, path: Path, overwrite: bool = False
    ) -> None:
        """Exports the resulting statistics to a csv file.

        Args:
            path: path to where the csv file should be written.
            overwrite: set to `True` to enable overwriting an existing file
        """
        if path.is_file() and not overwrite:
            raise FileExistsError(f"Expected that there was no file at {path}.")
        with path.open(mode="w") as f:
            table_writer = csv.writer(f)
            table_writer.writerow(self._header())
            # NOTE: we could stream / compute them later instead of collecting them...
            self._write_per_session_statistics(table_writer)

    @staticmethod
    def _header() -> List[Text]:
        return ["sender_id", "session_idx", "marker", "statistic", "value"]

    def _write_overview(self, table_writer: WriteRow) -> None:
        special_sender_idx = self.ALL_SENDERS
        special_session_idx = self.ALL_SESSIONS
        self._write_row(
            table_writer=table_writer,
            sender_id=special_sender_idx,
            session_idx=special_session_idx,
            marker_name=self.NO_MARKER,
            statistic_name=self.STAT_NUM_SESSIONS,
            statistic_value=self.num_sessions,
        )
        for marker_name, count in self.count_if_applied_at_least_once.items():
            self._write_row(
                table_writer=table_writer,
                sender_id=special_sender_idx,
                session_idx=special_session_idx,
                marker_name=marker_name,
                statistic_name=self.STAT_NUM_SESSIONS_WHERE_APPLIES,
                statistic_value=count,
            )
            self._write_row(
                table_writer=table_writer,
                sender_id=special_sender_idx,
                session_idx=special_session_idx,
                marker_name=marker_name,
                statistic_name=self.STAT_PERCENTAGE_SESSIONS_WHERE_APPLIES,
                statistic_value=(count / self.num_sessions * 100)
                if self.num_sessions
                else 100.0,
            )

    def _write_overall_statistics(self, table_writer: WriteRow) -> None:
        for marker_name, num_list in self.num_preceding_user_turns_collected.items():
            for statistic_name, value in compute_statistics(num_list).items():
                MarkerStatistics._write_row(
                    table_writer=table_writer,
                    sender_id=self.ALL_SENDERS,
                    session_idx=self.ALL_SESSIONS,
                    marker_name=marker_name,
                    statistic_name=self._add_num_user_turns_str_to(statistic_name),
                    statistic_value=value,
                )

    def _write_per_session_statistics(self, table_writer: WriteRow) -> None:
        for marker_name, collected_statistics in self.session_results.items():
            for statistic_name, values in collected_statistics.items():
                self._write_per_session_statistic(
                    table_writer=table_writer,
                    marker_name=marker_name,
                    statistic_name=statistic_name,
                    session_identifiers=self.session_identifier,
                    values=values,
                )

    @staticmethod
    def _write_per_session_statistic(
        table_writer: WriteRow,
        marker_name: Text,
        statistic_name: Text,
        session_identifiers: List[Tuple[Text, int]],
        values: List[Union[float, int]],
    ) -> None:
        for record_idx, (sender_id, session_idx) in enumerate(session_identifiers):
            MarkerStatistics._write_row(
                table_writer=table_writer,
                sender_id=sender_id,
                session_idx=session_idx,
                marker_name=marker_name,
                statistic_name=MarkerStatistics._add_num_user_turns_str_to(
                    statistic_name
                ),
                statistic_value=values[record_idx],
            )

    @staticmethod
    def _write_row(
        table_writer: WriteRow,
        sender_id: Text,
        session_idx: Union[int, float],
        marker_name: Text,
        statistic_name: Text,
        statistic_value: Union[int, float],
    ) -> None:
        if isinstance(statistic_value, int):
            value_str = str(statistic_value)
        elif np.isnan(statistic_value):
            value_str = str(np.nan)
        else:
            value_str = np.round(statistic_value, 3)
        table_writer.writerow(
            [
                str(item)
                for item in [
                    sender_id,
                    session_idx,
                    marker_name,
                    statistic_name,
                    value_str,
                ]
            ]
        )
