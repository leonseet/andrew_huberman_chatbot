import chromadb
from llama_index.vector_stores import ChromaVectorStore
from libs import configs
from chromadb.utils import embedding_functions
import os


def initialize_chroma():
    """
    Initializes the Chroma DB by creating the necessary components and returning the vector store.

    Returns:
        vector_store (ChromaVectorStore): The vector store used for storing and retrieving vectors.
    """
    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.getenv("HUGGINGFACE_TOKEN"), model_name=configs.EMB_MODEL
    )
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(
        name=configs.CONTEXT_COLLECTION, embedding_function=huggingface_ef
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store
