import json

inputFiles = [
    '2017-06-custom-intent-engines/GetWeather/train_GetWeather_full.json',
    '2017-06-custom-intent-engines/AddToPlaylist/train_AddToPlaylist_full.json',
    '2017-06-custom-intent-engines/BookRestaurant/train_BookRestaurant_full.json',
    '2017-06-custom-intent-engines/GetWeather/train_GetWeather_full.json',
    '2017-06-custom-intent-engines/PlayMusic/train_PlayMusic_full.json',
    '2017-06-custom-intent-engines/RateBook/train_RateBook_full.json',
    '2017-06-custom-intent-engines/SearchCreativeWork/train_SearchCreativeWork_full.json',
    '2017-06-custom-intent-engines/SearchScreeningEvent/train_SearchScreeningEvent_full.json',
]

for inputFile in inputFiles:
    print('Now converting: ' + inputFile)

    with open(inputFile, 'r', encoding='utf8') as f:
        data = json.loads(f.read())

    for intent in data:
        examples = data[intent]
        outputExamples = []

        for example in examples:
            example = example['data']

            text = ''
            textLength = 0
            entities = []
            exampleLength = len(example) - 1

            for index, phrase in enumerate(example):
                # print(index, exampleLength)

                if index != exampleLength:  # or (index == exampleLength and phrase['text'] != ' '):
                    text += phrase['text']
                    newLength = len(text)

                elif phrase['text'] != ' ':
                    phrase['text'] = phrase['text'].rstrip()
                    text += phrase['text']
                    newLength = len(text)

                if 'entity' in phrase:
                    entities.append({
                        "start": textLength,
                        "end": newLength,
                        "value": phrase['text'],
                        "entity": phrase['entity']
                    })

                textLength = newLength

            outputExamples.append({
                "text": text,
                "intent": intent,
                "entities": entities
            })

        outputRASA = {
            "rasa_nlu_data": {
                "common_examples": outputExamples
            }
        }
        outputFile = open(intent + '.json', 'w', encoding='utf8')
        outputFile.write(json.dumps(outputRASA, sort_keys=False, indent=2))
        outputFile.close()
