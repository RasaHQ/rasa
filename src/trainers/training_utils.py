import os
import json
import rasa_nlu


def relative_normpath(file, path):
    if file is not None:
        return os.path.normpath(os.path.relpath(file, path))
    else:
        return None


def write_training_metadata(output_folder, timestamp, backend_name,
                            language_name, additional_metadata, feature_file=None):
    metadata = additional_metadata.copy()

    metadata.update({
        "trained_at": timestamp,
        "backend": backend_name,
        "feature_extractor": relative_normpath(feature_file, output_folder),
        "language_name": language_name,
        "rasa_nlu_version": rasa_nlu.__version__
    })

    with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4))
