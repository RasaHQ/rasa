from rasa_nlu.components import Component


DUCKLING_PROCESSING_MODES = ["replace", "append"]


class DucklingExtractor(Component):
    name = "ner_duckling"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    def __init__(self, duckling=None):
        self.duckling = duckling

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> str

        return cls.name + "-" + model_metadata.language

    def pipeline_init(self, language, duckling_processing_mode):
        from duckling import DucklingWrapper

        if duckling_processing_mode not in DUCKLING_PROCESSING_MODES:
            raise ValueError("Invalid duckling processing mode. Got '{}'. Allowed: {}".format(
                duckling_processing_mode, ", ".join(DUCKLING_PROCESSING_MODES)))

        # If fine tuning is disabled, we do not need to load the spacy entity model
        if self.duckling is None:
            self.duckling = DucklingWrapper(language=language + "$core")  # languages in duckling are eg "de$core"

    def process(self, text, entities, duckling_processing_mode):
        if self.duckling is not None:
            parsed = self.duckling.parse(text)
            for duckling_match in parsed:
                for entity in entities:
                    if entity["start"] == duckling_match["start"] and entity["end"] == duckling_match["end"]:
                        entity["normalised"] = duckling_match["value"]["value"]
                        break
                else:
                    if duckling_processing_mode == "append":
                        # Duckling will retrieve multiple entities, even if they overlap..
                        # So as a compromise we'll only use the 'largest' ones in terms of sentence coverage
                        entities.append({
                            "entity": duckling_match["dim"],
                            "value": duckling_match["value"]["value"],
                            "start": duckling_match["start"],
                            "end": duckling_match["end"],
                        })

        return {
            "entities": entities
        }
