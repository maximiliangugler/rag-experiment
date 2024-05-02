import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
from get_prompt_embedding_sfr import get_embedding_function_sfr
from get_embedding_function_oai import get_embedding_function_oai

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():

    global ACTIVE_EMBEDDING_FUNCTION
    ACTIVE_EMBEDDING_FUNCTION = get_embedding_function()

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-openai-embedding", action="store_true", help="Use OpenAI Embedding Model.")
    parser.add_argument("--sfr-embedding", action="store_true", help="Uses the special prompt embedding needed for sfr.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    if args.use_openai_embedding:
        print("✨ Using OpenAI Embedding Model")
        ACTIVE_EMBEDDING_FUNCTION = get_embedding_function_oai()

    if args.sfr_embedding:
        print("✨ Using SFR Embedding")
        ACTIVE_EMBEDDING_FUNCTION = get_embedding_function_sfr("Given the following query, retrieve relevant passages that answer this query:", query_text)

    query_rag(query_text)






def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=ACTIVE_EMBEDDING_FUNCTION)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=4)
    print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = Ollama(model="mistral")
    print("ℹ️ Generation model is: ", model.model)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()