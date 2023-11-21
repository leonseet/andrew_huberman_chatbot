import chromadb
from llama_index import (
    ServiceContext,
)
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.storage.storage_context import StorageContext
from libs import configs


def initialize_chroma():
    """
    Initializes the Chroma DB by creating the necessary components and returning the storage and service contexts.

    Returns:
        storage_context (StorageContext): The storage context for the Chroma DB.
        service_context (ServiceContext): The service context for the Chroma DB.
    """
    embed_model = HuggingFaceEmbedding(model_name=configs.EMB_MODEL)
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(
        configs.CONTEXT_COLLECTION
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # service_context = ServiceContext.from_defaults(embed_model=embed_model)

    return vector_store
