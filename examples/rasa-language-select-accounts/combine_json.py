import json
import glob

result = []
# for f in glob.glob("*.json"):
#     print(f)
#     with open(f, "rb") as infile:
#         result.append(json.load(infile))

with open("account_mling.json") as fp:
    base_kb = json.load(fp)

with open("chitchat_mling.json") as fp:
    to_appned = json.load(fp)
    for key in to_appned:
        base_kb[key] = to_appned[key]

with open("multilingual_responses.json", "w") as outfile:
    json.dump(base_kb, outfile, indent=4)
