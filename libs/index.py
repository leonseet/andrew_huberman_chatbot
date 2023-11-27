from llama_index import ServiceContext
from llama_index.prompts import PromptTemplate
from llama_index.vector_stores import ChromaVectorStore
from llama_index.evaluation import DatasetGenerator
from llama_index.extractors import BaseExtractor
from llama_index.postprocessor import KeywordNodePostprocessor
from llama_index.schema import Node, NodeWithScore

import chromadb
from libs import configs
from chromadb.utils import embedding_functions
import os
from typing import List, Dict
from libs.query import get_llm


def initialize_chroma_vector_store(collection_name: str = configs.CONTEXT_COLLECTION):
    """
    Initializes the Chroma vector store by creating a collection and embedding function.

    Args:
        collection_name (str, optional): Name of the collection. Defaults to configs.CONTEXT_COLLECTION.

    Returns:
        ChromaVectorStore: The initialized Chroma vector store.
    """
    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.getenv("HUGGINGFACE_TOKEN"), model_name=configs.EMB_MODEL
    )
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=huggingface_ef,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store


def initialize_chroma_collection(collection_name: str = configs.CONTEXT_COLLECTION):
    """
    Initializes a Chroma collection with the given collection name.

    Args:
        collection_name (str): The name of the collection to initialize. Defaults to the value of `configs.CONTEXT_COLLECTION`.

    Returns:
        chroma_collection: The initialized Chroma collection.
    """
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
    )

    return chroma_collection


def postprocess(text):
    postprocessor = KeywordNodePostprocessor(
        # required_keywords=["?"],
        exclude_keywords=[
            "Inside Tracker",
            "premium channel",
            "podcast",
            "live events",
        ],
    )
    nodes = [NodeWithScore(node=Node(text=text), score=0)]
    nodes = postprocessor.postprocess_nodes(nodes)
    nodes = [node.text for node in nodes]
    return nodes[0] if nodes else None


class CustomQuestionExtractor(BaseExtractor):
    def extract(self, nodes) -> List[Dict]:
        num_questions_per_chunk = 2

        llm = get_llm("gpt-3.5-turbo")

        service_context = ServiceContext.from_defaults(
            llm=llm,
        )

        template = (
            "Context information is below."
            "\n---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given the context information and no prior knowledge, generate only scientific questions based on the below query. Generate <None> if no scientific question can be generated.\n"
            "{query_str}\n"
        )
        text_question_template = PromptTemplate(template)

        data_generator = DatasetGenerator(
            nodes=nodes,
            service_context=service_context,
            num_questions_per_chunk=num_questions_per_chunk,
            text_question_template=text_question_template,
            show_progress=False,
        )

        eval_questions = data_generator.generate_questions_from_nodes()

        metadata_list = []
        for i in range(0, len(eval_questions), num_questions_per_chunk):
            metadata_list.append(
                {
                    "questions_this_excerpt_can_answer": str(
                        [
                            postprocess(eval_questions[i]),
                            postprocess(eval_questions[i + 1]),
                        ]
                    )
                }
            )

        return metadata_list
