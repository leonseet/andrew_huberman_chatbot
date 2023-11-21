import json
from tqdm import tqdm
from transformers import LlamaTokenizerFast
from libs.index import initialize_chroma
from dotenv import load_dotenv

from llama_index.node_parser import TokenTextSplitter
from llama_index import (
    Document,
)
from llama_index.ingestion import IngestionPipeline

load_dotenv()

INPUT_FILE_PATH = "data/andrew_huberman_episodes_processed.json"

# Load the data
with open(INPUT_FILE_PATH, "r") as file:
    data = json.load(file)

# Initialize the tokenizer and splitter
tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
token_splitter = TokenTextSplitter(
    chunk_size=512, chunk_overlap=20, separator=" ", tokenizer=tokenizer
)

# Initialize the Chroma DB
vector_store = initialize_chroma()

for d in tqdm(data, total=len(data)):
    transcripts = d["transcripts"]
    if not transcripts:
        continue
    episode_metadata = {
        "episode_url": d["url"],
        "episode_created": d["created"],
        "episode_title": d["title"],
        "episode_description": d["description"],
        "episode_topics": str(d["topics"]),
        "episode_guest": str(d["guest"]),
        "episode_youtube": d["youtube"],
        "episode_length": d["episode_length"],
    }
    for t in transcripts:
        transcript_metadata = {
            "timestamp_start": t["start"],
            "timestamp_end": t["end"],
            "timestamp_sentencepiece_token_length": t["sentencepiece_token_length"],
            "timestamp_title": t["desc"],
        }
        print({**episode_metadata, **transcript_metadata})
        doc = Document(
            text=t["transcript"],
            extra_info={**episode_metadata, **transcript_metadata},
            excluded_embed_metadata_keys=[
                "episode_url",
                "episode_youtube",
                "episode_description",
                "timestamp_start",
                "timestamp_end",
                "timestamp_sentencepiece_token_length",
            ],
            excluded_llm_metadata_keys=[
                "timestamp_start",
                "timestamp_end",
                "timestamp_sentencepiece_token_length",
            ],
        )

        if t["sentencepiece_token_length"] > 512:
            # nodes = token_splitter.get_nodes_from_documents([doc])
            pipeline = IngestionPipeline(
                transformations=[
                    token_splitter,
                ],
                vector_store=vector_store,
            )
            pipeline.run(documents=[doc])
            # print(nodes)
            # index = VectorStoreIndex.from_documents(
            #     nodes, storage_context=storage_context, service_context=service_context
            # )

        else:
            # index = VectorStoreIndex.from_documents(
            #     [doc], storage_context=storage_context, service_context=service_context
            # )
            pipeline = IngestionPipeline(
                vector_store=vector_store,
            )
            pipeline.run(documents=[doc])
