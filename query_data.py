import os
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
from get_prompt_embedding_sfr import get_embedding_function_sfr
from get_embedding_function_oai import get_embedding_function_oai

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Context information is below:
---

{context}

---

Given only the context information above and no other knowledge, answer this qusetion:
{question}
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
        print("🌐 Using OpenAI Embedding Model")
        ACTIVE_EMBEDDING_FUNCTION = get_embedding_function_oai()

    if args.sfr_embedding:
        print("✨ Using SFR Embedding")
        ACTIVE_EMBEDDING_FUNCTION = get_embedding_function_sfr("Given the following query, retrieve relevant passages that answer this question:", query_text)

    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=ACTIVE_EMBEDDING_FUNCTION)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=4)
     # print(results)


    context_texts = []
    for doc, _score in results:
        source = doc.metadata.get("id", None)
        filename = os.path.basename(source)
        filename = filename.split(':', 1)[0]
        context_text = f"From File '{filename}': \n{doc.page_content}"
        context_texts.append(context_text)

    context_text = "\n\n---\n\n".join(context_texts)
     

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("PROMPT: ", prompt)

    model = Ollama(model="llama3:8b-instruct-q8_0")
    print("ℹ️ Generation model is: ", model.model)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()