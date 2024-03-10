from rasa.shared.core.domain import SessionConfig
from rasa.shared.importers.importer import TrainingDataImporter


def test_merge_domain_with_custom_session_config_and_no_session_config():
    expected_session_expiration_time = 0
    expected_carry_over_slots = False

    config_path = (
        "data/test_domains/"
        "test_domain_files_with_no_session_config_and_custom_session_config/config.yml"
    )
    domain_path = (
        "data/test_domains/"
        "test_domain_files_with_no_session_config_and_custom_session_config/domain.yml"
    )
    training_data_paths = [
        "data/test_domains/"
        "test_domain_files_with_no_session_config_and_custom_session_config/data"
    ]
    file_importer = TrainingDataImporter.load_from_config(
        config_path, domain_path, training_data_paths
    )

    domain = file_importer.get_domain()

    assert (
        domain.session_config.session_expiration_time
        != SessionConfig.default().session_expiration_time
    )
    assert (
        domain.session_config.carry_over_slots
        != SessionConfig.default().carry_over_slots
    )

    assert (
        domain.session_config.session_expiration_time
        == expected_session_expiration_time
    )
    assert domain.session_config.carry_over_slots == expected_carry_over_slots
