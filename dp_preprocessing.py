import json
from transformers import LlamaTokenizerFast
import huggingface_hub
import os

huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"), new_session=False)


INPUT_FILE_PATH = "data/andrew_huberman_episodes.json"
OUTPUT_FILE_PATH = "data/andrew_huberman_episodes_processed.json"
CHAT_MODEL = "meta-llama/Llama-2-7b-chat-hf"

with open(INPUT_FILE_PATH, "r") as file:
    data = json.load(file)


tokenizer = LlamaTokenizerFast.from_pretrained(CHAT_MODEL)
for episode in data:
    transcripts = episode["transcripts"]
    if transcripts:
        # Get token length of each transcript
        for item in transcripts:
            tokenized_chunk = tokenizer.encode(item["transcript"])
            item["sentencepiece_token_length"] = len(tokenized_chunk)

        # Get episode length
        first_transcript = transcripts[0]
        last_transcript = transcripts[-1]
        episode_length = last_transcript["end"] - first_transcript["start"]
        episode["episode_length"] = episode_length


with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
