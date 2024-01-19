from pathlib import Path

from rasa.shared.core.domain import Domain
from rasa.shared.utils.cli import print_error_and_exit

from rasa.markers.marker import OrMarker
from rasa.markers.marker_base import Marker


def validate_marker_file(domain: Domain, markers_path: Path) -> None:
    markers = Marker.from_path(markers_path)
    validate_markers(domain, markers)


def validate_markers(domain: Domain, markers: OrMarker) -> None:
    """Validate markers."""
    if domain and not markers.validate_against_domain(domain):
        print_error_and_exit(
            "Validation errors were found in the markers definition. "
            "Please see errors listed above and fix before running again."
        )
