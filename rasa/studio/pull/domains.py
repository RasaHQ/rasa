import logging
from pathlib import Path

from rasa.shared.core.domain import (
    Domain,
)
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.studio.constants import STUDIO_DOMAIN_FILENAME

logger = logging.getLogger(__name__)


def merge_domain(
    data_from_studio: TrainingDataImporter,
    data_local: TrainingDataImporter,
    domain_path: Path,
) -> None:
    """Merges the domain from Rasa Studio with the local domain.

    Args:
        data_from_studio: The training data importer for the Rasa Studio domain.
        data_local: The training data importer for the local domain.
        domain_path: The path to the domain file or directory.
    """
    if domain_path.is_file():
        all_local_domain_files = [str(domain_path)]
        domain_path = domain_path.parent
    else:
        all_local_domain_files = data_local.get_domain_files([str(domain_path)])

    studio_domain_file_path = domain_path / STUDIO_DOMAIN_FILENAME

    # leftover_domain represents the items in the studio
    # domain that are not in the local domain
    leftover_domain = data_from_studio.get_user_domain()
    for file_path in all_local_domain_files:
        if file_path == str(studio_domain_file_path):
            # we need to exclude the studio domain file from the merge,
            # since we want to dump ALL the remaining items to this path
            # after the merge. if we include it here, we will remove the existing
            # items from the leftover domain and after this loop we will
            # overwrite the studio domain file with the remaining items in
            # the leftover domain - this means we loose the items that were
            # in the studio domain file before we started the merge.
            continue

        # For each local domain file, we do a partial merge
        local_domain = Domain.from_file(str(file_path))
        updated_local_domain = local_domain.partial_merge(leftover_domain)

        # If this partial merge made changes, persist them
        if local_domain != updated_local_domain:
            updated_local_domain.persist(file_path)

        # Remove every item now present in updated_local_domain from leftover_domain
        leftover_domain = leftover_domain.difference(updated_local_domain)

    # If there are still items in leftover_domain, persist them
    if not leftover_domain.is_empty():
        leftover_domain.persist(studio_domain_file_path)
