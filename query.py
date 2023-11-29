from libs.index import initialize_chroma_vector_store
from libs import configs
from dotenv import load_dotenv
import os

from llama_index.llms.anyscale import Anyscale
from llama_index.llms import OpenAI
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
    ServiceContext,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.schema import MetadataMode
from llama_index.embeddings import HuggingFaceEmbedding

load_dotenv()

QUESTION = "How does the hormone ghrelin affect hunger and appetite regulation?"

# llm = Anyscale(
#     model="meta-llama/Llama-2-70b-chat-hf",
#     api_key=os.getenv("ANYSCALE_API_KEY"),
#     temperature=0.1,
#     max_tokens=512,
# )
# llm = OpenAI(model="gpt-4", temperature=0.1, max_tokens=512)

# configure service context
llm = OpenAI(model=configs.QA_MODEL, temperature=0.1, max_tokens=1000)
embed_model = HuggingFaceEmbedding(model_name=configs.EMB_MODEL)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# initialize index
vector_store = initialize_chroma_vector_store()
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, service_context=service_context
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    service_context=service_context, verbose=True
)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
    verbose=False,
)

# retrieve
nodes = retriever.retrieve(QUESTION)
for node in nodes:
    node.node.excluded_llm_metadata_keys = [
        "episode_description",
        "timestamp_start",
        "timestamp_end",
        "timestamp_sentencepiece_token_length",
    ]

# query
response = response_synthesizer.synthesize(query=QUESTION, nodes=nodes)

for node in response.source_nodes:
    print(node.node_id)
    # print(node.metadata)
    print(node.get_content(metadata_mode=MetadataMode.LLM))
    print(node.score)
    print("-" * 20)

print(">>>", response)
# print(">>>", response.source_nodes)
