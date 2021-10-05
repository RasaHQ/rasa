import textwrap
from pathlib import Path
import rasa.shared.utils.io
import rasa.core.migrate
from rasa.shared.core.domain import Domain


def test_migrate_domain_format(tmp_path: Path):
    original_domain = textwrap.dedent(
        """
        intents:
        - greet
        - affirm
        - inform
        entities:
        - city
        - name
        slots:
          location:
            type: text
            influence_conversation: false
          name:
            type: text
            influence_conversation: false
            auto_fill: false
          email:
            type: text
            influence_conversation: false
        forms:
           booking_form:
               location:
                 - type: from_entity
                   entity: city
               email:
                 - type: from_text
                   intent: inform
        """
    )
    domain_file = tmp_path / "domain.yml"
    rasa.shared.utils.io.write_text_file(original_domain, domain_file)

    new_domain_file = tmp_path / "new_domain.yml"
    rasa.core.migrate.migrate_domain_format(domain_file, new_domain_file)

    domain = Domain.from_path(new_domain_file)
    assert domain
