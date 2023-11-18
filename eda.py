import json

INPUT_FILE_PATH = "data/andrew_huberman_episodes.json"

with open(INPUT_FILE_PATH, "r") as file:
    data = json.load(file)

print(data[0]["transcripts"][0])
