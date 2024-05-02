from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embedding_function_oai():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="sk-proj-0AeNS5Mz0hSZ4lhqgCXeT3BlbkFJir9RcfcQHDjPmB7aZpe5"
    )
    return embeddings