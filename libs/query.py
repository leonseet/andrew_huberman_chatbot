from llama_index.llms.anyscale import Anyscale
from llama_index.llms import OpenAI
import os


TEMPERATURE = 0.1
MAX_TOKENS = 512


def get_llm(model_name: str):
    if model_name in ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
        return OpenAI(model=model_name, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    else:
        return Anyscale(
            model=model_name,
            api_key=os.getenv("ANYSCALE_API_KEY"),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
