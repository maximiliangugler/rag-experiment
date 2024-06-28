from langchain_openai import OpenAIEmbeddings


def get_embedding_function_oai():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key="..."
    )
    return embeddings