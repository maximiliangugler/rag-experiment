from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embedding_function_oai():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    return embeddings