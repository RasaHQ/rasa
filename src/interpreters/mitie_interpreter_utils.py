import re


def get_entities(text, tokens, extractor, featurizer):
    ents = []
    if extractor:
        entities = extractor.extract_entities(tokens, featurizer.feature_extractor)
        for e in entities:
            _range = e[0]
            _regex = u"\s*".join(re.escape(tokens[i]) for i in _range)
            expr = re.compile(_regex)
            m = expr.search(text)
            start, end = m.start(), m.end()
            entity_value = text[start:end]
            ents.append({
                "entity": e[1],
                "value": entity_value,
                "start": start,
                "end": end
            })

    return ents
