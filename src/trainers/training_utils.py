import os
import json
import rasa_nlu


def write_training_metadata(output_folder, timestamp, data_file, backend_name,
                            language_name, intent_file, entity_file,
                            feature_file=None):
    metadata = {
        "trained_at": timestamp,
        "training_data": data_file,
        "backend": backend_name,
        "intent_classifier": intent_file,
        "entity_extractor": entity_file,
        "feature_extractor": feature_file,
        "language_name": language_name,
        "rasa_nlu_version": rasa_nlu.__version__
    }

    with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4))
