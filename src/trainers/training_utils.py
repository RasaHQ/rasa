import os
import json
import rasa_nlu


def __relative_normpath(file, path):
    if file is not None:
        return os.path.normpath(os.path.relpath(file, path))
    else:
        return None


def write_training_metadata(output_folder, timestamp, data_file, backend_name,
                            language_name, intent_file, entity_file,
                            entity_synonyms_file=None, feature_file=None):

    metadata = {
        "trained_at": timestamp,
        "training_data": os.path.basename(data_file),
        "backend": backend_name,
        "intent_classifier": __relative_normpath(intent_file, output_folder),
        "entity_extractor": __relative_normpath(entity_file, output_folder),
        "entity_synonyms": __relative_normpath(entity_synonyms_file, output_folder),
        "feature_extractor": __relative_normpath(feature_file, output_folder),
        "language_name": language_name,
        "rasa_nlu_version": rasa_nlu.__version__
    }

    with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4))
