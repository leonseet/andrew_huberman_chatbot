import json
import uuid

INPUT_FILE_PATH = "data/andrew_huberman_episodes_processed.json"
OUTPUT_FILE_PATH = "data/andrew_huberman_episodes_processed_uuid.json"

with open(INPUT_FILE_PATH, "r") as f:
    data = json.load(f)

for episode in data:
    transcripts = episode["transcripts"]
    if not transcripts:
        continue
    for t in transcripts:
        t["uuid"] = str(uuid.uuid4())

with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
