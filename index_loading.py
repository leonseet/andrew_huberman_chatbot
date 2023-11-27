import json
from tqdm import tqdm
from transformers import LlamaTokenizerFast
from libs.index import initialize_chroma_vector_store, CustomQuestionExtractor
from dotenv import load_dotenv
from libs import configs
from libs.query import get_llm

from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import TokenTextSplitter
from llama_index import (
    Document,
)
from llama_index.ingestion import IngestionPipeline

load_dotenv()

INPUT_FILE_PATH = "data/andrew_huberman_episodes_processed_small.json"

# Load the data
with open(INPUT_FILE_PATH, "r") as file:
    data = json.load(file)

# Initialize splitter
embed_model = HuggingFaceEmbedding(model_name=configs.EMB_MODEL)
llm = get_llm("gpt-3.5-turbo")
text_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    separator=" ",
)

# Initialize the Chroma DB
vector_store = initialize_chroma_vector_store()

docs = []

for d in tqdm(data, total=len(data), desc="Processing into list of Document"):
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
                "episode_description",
                "timestamp_start",
                "timestamp_end",
                "timestamp_sentencepiece_token_length",
            ],
        )

        docs.append(doc)

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        QuestionsAnsweredExtractor(questions=3, llm=llm),
        # CustomQuestionExtractor(),
        embed_model,
    ],
    vector_store=vector_store,
)
nodes = pipeline.run(documents=docs, show_progress=True)
