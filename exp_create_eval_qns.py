from libs.index import initialize_chroma_collection
from tqdm import tqdm
import re
import json

OUTPUT_FILE_PATH = "data/eval_qns_small.json"

chroma_collection = initialize_chroma_collection()
docs = chroma_collection.get(limit=9999, include=["metadatas", "documents"])

metadatas = docs["metadatas"]
documents = docs["documents"]
ids = docs["ids"]

eval_qns = []

for i in tqdm(range(len(metadatas)), total=len(metadatas)):
    metadata = metadatas[i]
    doc = documents[i]

    doc_id = ids[i]
    questions = metadata["questions_this_excerpt_can_answer"]
    questions = questions.split("\n")
    first_qns = questions[0]
    first_qns = first_qns.strip()
    first_qns = re.sub("1. ", "", first_qns)

    eval_qns.append(
        {
            "doc_id": doc_id,
            "qns": first_qns,
            "source_doc": doc,
        }
    )

with open(OUTPUT_FILE_PATH, "w") as f:
    json.dump(eval_qns, f, indent=4)
